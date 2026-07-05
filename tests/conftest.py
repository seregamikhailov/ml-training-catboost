"""Фикстуры: синтетический мини-датасет + прогон пайплайна по этапам.

Генерируем сырые данные той же структуры, что ждет пайплайн (недельные
parquet-шарды взаимодействий, метаданные пользователей/видео, npz с
контентными эмбеддингами), с "подсаженным" сигналом: у каждого видео есть
латентное качество q, от которого зависят и досмотры, и лайки. Благодаря
этому агрегатные фичи предсказательны, и на smoke-обучении модель обязана
обыгрывать случайное ранжирование — это и проверяем.

Фикстуры session-scope выстроены цепочкой:
    pipe_cfg (данные) -> prepared -> processed -> featured -> dataset -> trained
поэтому каждый этап пайплайна выполняется в тестовой сессии ровно один раз.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.common import load_config

N_USERS = 60
N_ITEMS = 120
N_AUTHORS = 25
EVENTS_PER_USER = 40
EMB_DIM = 8
N_SHARDS = 3
SEED = 7

RARE_USER_ID = 999_999   # 1 событие -> должен отфильтроваться
RARE_ITEM_ID = 888_888   # 1 событие -> должен отфильтроваться


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_raw_data(raw_dir: Path, rng: np.random.Generator) -> None:
    """Пишет недельные шарды взаимодействий, метаданные и эмбеддинги —
    в той же структуре каталогов, что и реальные данные."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "metadata").mkdir(exist_ok=True)

    user_ids = np.arange(1, N_USERS + 1)
    item_ids = np.arange(101, 101 + N_ITEMS)
    author_of_item = rng.integers(1, N_AUTHORS + 1, N_ITEMS)
    duration = rng.integers(5, 121, N_ITEMS)
    quality = rng.normal(0.0, 1.5, N_ITEMS)  # латентное качество видео

    # Эмбеддинг коррелирует с качеством: первая компонента ~ q.
    emb = rng.normal(0.0, 1.0, (N_ITEMS, EMB_DIM)).astype(np.float32)
    emb[:, 0] = quality.astype(np.float32)

    # --- события в глобальном временном порядке
    n_events = N_USERS * EVENTS_PER_USER
    ev_user = rng.choice(user_ids, n_events)
    ev_item_idx = rng.integers(0, N_ITEMS, n_events)
    q = quality[ev_item_idx]
    dur = duration[ev_item_idx]

    ratio = np.clip(_sigmoid(q + rng.normal(0, 0.5, n_events)) * 1.3, 0.0, 1.2)
    timespent = (dur * ratio).astype(np.float64)
    like = (rng.random(n_events) < _sigmoid(3 * q - 1)).astype(np.int8)
    dislike = (rng.random(n_events) < _sigmoid(-3 * q - 2)).astype(np.int8)
    share = (rng.random(n_events) < _sigmoid(2 * q - 3)).astype(np.int8)
    bookmark = (rng.random(n_events) < _sigmoid(2 * q - 3)).astype(np.int8)
    click_on_author = (rng.random(n_events) < 0.05).astype(np.int8)
    open_comments = (rng.random(n_events) < 0.05).astype(np.int8)

    inter = pl.DataFrame(
        {
            "user_id": ev_user,
            "item_id": item_ids[ev_item_idx],
            "timespent": timespent,
            "like": like,
            "dislike": dislike,
            "share": share,
            "bookmark": bookmark,
            "click_on_author": click_on_author,
            "open_comments": open_comments,
            "place": rng.choice(["feed", "search", "profile"], n_events),
            "platform": rng.choice(["ios", "android", "web"], n_events),
            "agent": rng.choice(["app", "mobile_web"], n_events),
        }
    )

    # Аномалии для проверки очистки: гигантский timespent и перекрут ratio.
    anomalies = pl.DataFrame(
        {
            "user_id": [int(user_ids[0])] * 3,
            "item_id": [int(item_ids[0])] * 3,
            "timespent": [10_000_000.0, 500.0, -5.0],  # фильтр / клип ratio / фильтр
            "like": [0, 0, 0], "dislike": [0, 0, 0], "share": [0, 0, 0],
            "bookmark": [0, 0, 0], "click_on_author": [0, 0, 0],
            "open_comments": [0, 0, 0],
            "place": ["feed"] * 3, "platform": ["ios"] * 3, "agent": ["app"] * 3,
        }
    ).with_columns(pl.col(c).cast(inter.schema[c]) for c in inter.columns)

    # Редкие user/item (единственное событие) — должны уйти по min-активности.
    rare = pl.DataFrame(
        {
            "user_id": [RARE_USER_ID],
            "item_id": [RARE_ITEM_ID],
            "timespent": [10.0],
            "like": [1], "dislike": [0], "share": [0], "bookmark": [0],
            "click_on_author": [0], "open_comments": [0],
            "place": ["feed"], "platform": ["ios"], "agent": ["app"],
        }
    ).with_columns(pl.col(c).cast(inter.schema[c]) for c in inter.columns)

    inter = pl.concat([inter, anomalies, rare])

    # Шардируем "по неделям" с сохранением порядка строк.
    inter_dir = raw_dir / "interactions"
    inter_dir.mkdir(exist_ok=True)
    for i, part in enumerate(np.array_split(np.arange(len(inter)), N_SHARDS)):
        p = inter_dir / f"week_{i:02d}.parquet"
        inter[int(part[0]) : int(part[-1]) + 1].write_parquet(p)

    users = pl.DataFrame(
        {
            "user_id": np.append(user_ids, RARE_USER_ID),
            "age": rng.integers(14, 60, N_USERS + 1),
            "gender": rng.choice(["m", "f"], N_USERS + 1),
            "geo": rng.choice(["msk", "spb", "nsk", "ekb"], N_USERS + 1),
        }
    )
    users.write_parquet(raw_dir / "metadata" / "users_metadata.parquet")

    items = pl.DataFrame(
        {
            "item_id": np.append(item_ids, RARE_ITEM_ID),
            "duration": np.append(duration, 30),
            "author_id": np.append(author_of_item, 1),
        }
    )
    items.write_parquet(raw_dir / "metadata" / "items_metadata.parquet")

    np.savez(raw_dir / "metadata" / "item_embeddings.npz",
             item_id=item_ids, embedding=emb)


@pytest.fixture(scope="session")
def pipe_cfg(tmp_path_factory) -> dict:
    """Реальный configs/pipeline.yaml с путями в tmp и выключенным ClearML."""
    cfg = copy.deepcopy(load_config("configs/pipeline.yaml"))
    root = tmp_path_factory.mktemp("ranking")
    cfg["clearml"]["enabled"] = False
    cfg["paths"]["raw_dir"] = str(root / "raw")
    cfg["paths"]["processed_dir"] = str(root / "processed")
    cfg["cleaning"]["min_user_interactions"] = 3
    cfg["cleaning"]["min_item_interactions"] = 2
    generate_raw_data(Path(cfg["paths"]["raw_dir"]), np.random.default_rng(SEED))
    return cfg


@pytest.fixture(scope="session")
def prepared(pipe_cfg) -> Path:
    from src import prepare_data

    prepare_data.main(pipe_cfg)
    return Path(pipe_cfg["paths"]["raw_dir"])


@pytest.fixture(scope="session")
def processed(pipe_cfg, prepared) -> Path:
    from src import preprocess

    preprocess.main(pipe_cfg)
    return Path(pipe_cfg["paths"]["processed_dir"])


@pytest.fixture(scope="session")
def featured(pipe_cfg, processed) -> Path:
    from src import features

    features.main(pipe_cfg)
    return processed


@pytest.fixture(scope="session")
def dataset(pipe_cfg, featured) -> Path:
    from src import build_dataset

    build_dataset.main(pipe_cfg)
    return featured / "dataset"


@pytest.fixture(scope="session")
def train_cfg(pipe_cfg, dataset, tmp_path_factory) -> dict:
    """configs/train.yaml, ужатый до smoke-размеров и CPU."""
    cfg = copy.deepcopy(load_config("configs/train.yaml"))
    cfg["clearml"]["enabled"] = False
    cfg["paths"]["dataset_dir"] = str(dataset)
    cfg["paths"]["models_dir"] = str(tmp_path_factory.mktemp("models"))
    cfg["model"].update(
        iterations=60,
        depth=4,
        task_type="CPU",
        verbose=0,
        metric_period=10,
        early_stopping_rounds=None,
    )
    cfg["model"].pop("devices", None)
    return cfg


@pytest.fixture(scope="session")
def trained(train_cfg) -> dict:
    from src import train

    train.main(train_cfg)
    return train_cfg
