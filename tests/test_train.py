"""Smoke-тест обучения CatBoost-ранкера и sanity-проверки качества модели.

Обучение идет на синтетике с подсаженным сигналом (label зависит от
латентного качества видео), поэтому обученная модель обязана заметно
обыгрывать случайное ранжирование — иначе пайплайн сломан.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from src.metrics import compute_ranking_metrics
from src.train import load_split


@pytest.fixture(scope="session")
def artifacts(trained) -> Path:
    return Path(trained["paths"]["models_dir"])


@pytest.fixture(scope="session")
def offline_metrics(artifacts) -> dict:
    with open(artifacts / "offline_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def model(artifacts):
    from catboost import CatBoostRanker

    m = CatBoostRanker()
    m.load_model(str(artifacts / "catboost_ranker.cbm"))
    return m


def test_artifacts_written(artifacts):
    for name in ("catboost_ranker.cbm", "offline_metrics.json",
                 "feature_importance.csv"):
        assert (artifacts / name).exists(), name


def test_offline_metrics_structure_and_ranges(trained, offline_metrics):
    ks = trained["eval"]["ks"]
    for split in ("val", "test"):
        metrics = offline_metrics[split]
        for k in ks:
            for m in ("ndcg", "map", "mrr", "hitrate", "recall", "precision"):
                assert f"{m}@{k}" in metrics, f"{split}/{m}@{k}"
        for name, value in metrics.items():
            if name != "n_groups":
                assert 0.0 <= value <= 1.0, f"{split}/{name}={value}"
        assert metrics["n_groups"] > 1


def test_model_beats_random_ranking(trained, model):
    """Ключевая проверка качества: NDCG@10 и GAUC модели против случайных
    скоров на val."""
    dataset_dir = Path(trained["paths"]["dataset_dir"])
    with open(dataset_dir / "dataset_meta.yaml", "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    X_val, y_val, g_val = load_split(dataset_dir, "val", meta)

    model_scores = model.predict(X_val)
    rng = np.random.default_rng(0)
    random_scores = rng.normal(size=len(y_val))

    ours = compute_ranking_metrics(y_val, model_scores, g_val, ks=(10,))
    rand = compute_ranking_metrics(y_val, random_scores, g_val, ks=(10,))

    assert ours["ndcg@10"] > rand["ndcg@10"] + 0.03, (ours, rand)
    assert ours["gauc"] > 0.55
    assert abs(rand["gauc"] - 0.5) < 0.1  # sanity самого бейзлайна


def test_predictions_are_not_constant(trained, model):
    dataset_dir = Path(trained["paths"]["dataset_dir"])
    with open(dataset_dir / "dataset_meta.yaml", "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    X_val, _, _ = load_split(dataset_dir, "val", meta)
    scores = model.predict(X_val)
    assert np.std(scores) > 1e-6


def test_feature_importance_covers_all_features(trained, artifacts):
    dataset_dir = Path(trained["paths"]["dataset_dir"])
    with open(dataset_dir / "dataset_meta.yaml", "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    fi = pl.read_csv(artifacts / "feature_importance.csv")
    n_features = len(meta["num_features"]) + len(meta["cat_features"])
    assert len(fi) == n_features
    # Важности не все нулевые — модель что-то выучила.
    importance_col = [c for c in fi.columns if c.lower() != "feature id"
                      and fi[c].dtype in (pl.Float64, pl.Float32)][0]
    assert fi[importance_col].abs().sum() > 0


def test_val_groups_suitable_for_ranking(trained):
    """В val есть группы с >= 2 объектами — иначе ранжирующие метрики
    вырождаются."""
    dataset_dir = Path(trained["paths"]["dataset_dir"])
    val = pl.read_parquet(dataset_dir / "val.parquet")
    group_sizes = val.group_by("user_id").len()
    assert (group_sizes["len"] >= 2).sum() > 0
