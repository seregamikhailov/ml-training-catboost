"""Этап 3. Построение фичей для ранжирования.

Все агрегаты считаются ТОЛЬКО на train-окне (первые train_frac событий по
глобальному временному порядку), чтобы не было утечки будущего в val/test.

Группы фичей:
- юзерские:      активность, средний watch_ratio, доли фидбека, разнообразие
                 авторов, средняя длительность просматриваемых видео + статика
                 (age, gender, geo — уже в interactions.parquet);
- айтемные:      популярность, средний watch_ratio, доли фидбека, duration,
                 агрегаты автора;
- юзер-айтемные: история пользователя с автором видео (ua_*), |duration -
                 средняя длительность у юзера| и косинус контентного
                 эмбеддинга видео с эмбеддинг-профилем пользователя
                 (среднее эмбеддингов его train-позитивов), считается на
                 torch (cuda / mps / cpu).

Выход: data/processed/features/*.parquet + split.yaml с границами сплита.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pyarrow as pa
import yaml

from src.common import (
    close_clearml,
    get_torch_device,
    init_clearml,
    load_config,
    log,
    resolve_path,
    setup_logging,
)
from src.preprocess import EVENT_ORDER, LABEL, WATCH_RATIO, load_manifest


def _rate_cols(feedback: list[str], prefix: str) -> list[pl.Expr]:
    """Средние по фидбек-колонкам -> доли лайков/дизлайков/шеров и т.д."""
    return [
        pl.col(f).cast(pl.Float32).mean().alias(f"{prefix}_{f}_rate") for f in feedback
    ]


def compute_split_bounds(interactions_path, split_cfg: dict) -> dict:
    q = pl.scan_parquet(interactions_path).select(
        train_end=pl.col(EVENT_ORDER).quantile(split_cfg["train_frac"]),
        val_end=pl.col(EVENT_ORDER).quantile(
            split_cfg["train_frac"] + split_cfg["val_frac"]
        ),
    ).collect()
    return {
        "train_end": int(q["train_end"][0]),
        "val_end": int(q["val_end"][0]),
    }


def build_aggregates(cfg: dict, interactions_path, features_dir, train_end: int) -> None:
    c = cfg["columns"]
    feedback = list(c["feedback"])
    train_lf = pl.scan_parquet(interactions_path).filter(
        pl.col(EVENT_ORDER) < train_end
    )

    user_feats = train_lf.group_by(c["user_id"]).agg(
        pl.len().alias("user_n_events"),
        pl.col(WATCH_RATIO).mean().alias("user_mean_watch_ratio"),
        pl.col(c["timespent"]).mean().alias("user_mean_timespent"),
        pl.col(c["duration"]).mean().alias("user_mean_duration"),
        pl.col(c["author_id"]).n_unique().alias("user_n_authors"),
        *_rate_cols(feedback, "user"),
    )
    user_feats.sink_parquet(features_dir / "user_features.parquet")

    item_feats = train_lf.group_by(c["item_id"]).agg(
        pl.len().alias("item_n_views"),
        pl.col(WATCH_RATIO).mean().alias("item_mean_watch_ratio"),
        pl.col(LABEL).mean().alias("item_mean_label"),
        *_rate_cols(feedback, "item"),
    )
    item_feats.sink_parquet(features_dir / "item_features.parquet")

    author_feats = train_lf.group_by(c["author_id"]).agg(
        pl.len().alias("author_n_views"),
        pl.col(c["item_id"]).n_unique().alias("author_n_items"),
        pl.col(WATCH_RATIO).mean().alias("author_mean_watch_ratio"),
        *_rate_cols(feedback, "author"),
    )
    author_feats.sink_parquet(features_dir / "author_features.parquet")

    ua_feats = train_lf.group_by([c["user_id"], c["author_id"]]).agg(
        pl.len().alias("ua_n_events"),
        pl.col(WATCH_RATIO).mean().alias("ua_mean_watch_ratio"),
        pl.col(LABEL).mean().alias("ua_mean_label"),
    )
    ua_feats.sink_parquet(features_dir / "ua_features.parquet")
    log.info("Агрегатные фичи записаны в %s", features_dir)


# ---------------------------------------------------------------------------
# Эмбеддинг-фича: cos(профиль пользователя, эмбеддинг видео)
# ---------------------------------------------------------------------------

def _load_item_embeddings(cfg: dict, manifest: dict, needed_items: pl.Series):
    """Возвращает (ids: np.ndarray отсортированные, emb: np.ndarray [n, dim]),
    только для айтемов из выборки.

    Эмбеддинги лежат в npz-файле (массивы item_id и embedding);
    parquet с list-колонкой поддержан как fallback для других форматов."""
    c = cfg["columns"]
    files = manifest.get("embeddings") or manifest.get("items") or []
    if not files:
        return None, None

    npz_files = sorted(f for f in files if f.endswith(".npz"))
    if npz_files:
        ids_parts, emb_parts = [], []
        for f in npz_files:
            data = np.load(f)
            ids_parts.append(data["item_id"])
            emb_parts.append(data["embedding"])
        ids = np.concatenate(ids_parts)
        emb = np.concatenate(emb_parts)
        mask = np.isin(ids, needed_items.to_numpy())
        ids, emb = ids[mask], emb[mask]
        order = np.argsort(ids)
        ids, emb = ids[order], np.ascontiguousarray(emb[order], dtype=np.float32)
    else:
        lf = pl.scan_parquet(sorted(files))
        if c["embedding"] not in lf.collect_schema().names():
            log.warning("Колонка '%s' не найдена в файлах эмбеддингов", c["embedding"])
            return None, None
        df = (
            lf.select(c["item_id"], c["embedding"])
            .join(needed_items.to_frame(c["item_id"]).lazy(), on=c["item_id"], how="semi")
            .collect()
            .sort(c["item_id"])
        )
        ids = df[c["item_id"]].to_numpy()
        emb = np.asarray(df[c["embedding"]].to_list(), dtype=np.float32)

    # Matryoshka-усечение: как в Quick Start датасета, первые N компонент.
    trunc_dim = cfg["features"].get("embedding_dim")
    if trunc_dim:
        emb = np.ascontiguousarray(emb[:, :trunc_dim])
    # L2-нормировка: дальше косинус — это просто скалярное произведение.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)
    log.info("Эмбеддинги: %d айтемов, dim=%d", emb.shape[0], emb.shape[1])
    return ids, emb


def _map_ids(sorted_ids: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Индексы query в sorted_ids; -1 если id отсутствует."""
    idx = np.searchsorted(sorted_ids, query)
    idx = np.clip(idx, 0, len(sorted_ids) - 1)
    valid = sorted_ids[idx] == query
    return np.where(valid, idx, -1)


def build_embedding_feature(cfg: dict, interactions_path, features_dir, train_end: int) -> bool:
    import torch

    c = cfg["columns"]
    f_cfg = cfg["features"]
    chunk = int(f_cfg["emb_chunk_size"])
    manifest = load_manifest(cfg)

    needed_items = (
        pl.scan_parquet(interactions_path).select(c["item_id"]).unique().collect()
    )[c["item_id"]]
    item_ids, item_emb = _load_item_embeddings(cfg, manifest, needed_items)
    if item_ids is None:
        return False

    device = get_torch_device()
    log.info("Считаем эмбеддинг-фичу на устройстве: %s", device)
    emb_t = torch.from_numpy(item_emb).to(device)
    dim = emb_t.shape[1]

    user_ids = np.sort(
        pl.scan_parquet(interactions_path)
        .select(c["user_id"])
        .unique()
        .collect()[c["user_id"]]
        .to_numpy()
    )
    profiles = torch.zeros((len(user_ids), dim), device=device)
    counts = torch.zeros(len(user_ids), device=device)

    dataset = pads.dataset(str(interactions_path))

    # Проход 1: аккумулируем профили по train-позитивам.
    pos_thr = float(f_cfg["profile_pos_label"])
    flt = (pads.field(EVENT_ORDER) < train_end) & (pads.field(LABEL) >= pos_thr)
    for batch in dataset.to_batches(
        columns=[c["user_id"], c["item_id"]], filter=flt, batch_size=chunk
    ):
        u = _map_ids(user_ids, batch[c["user_id"]].to_numpy())
        i = _map_ids(item_ids, batch[c["item_id"]].to_numpy())
        mask = (u >= 0) & (i >= 0)
        if not mask.any():
            continue
        u_t = torch.from_numpy(u[mask].astype(np.int64)).to(device)
        i_t = torch.from_numpy(i[mask].astype(np.int64)).to(device)
        profiles.index_add_(0, u_t, emb_t[i_t])
        counts.index_add_(0, u_t, torch.ones(len(u_t), device=device))

    has_profile = counts > 0
    profiles[has_profile] /= counts[has_profile].unsqueeze(1)
    profiles = torch.nn.functional.normalize(profiles, dim=1, eps=1e-12)
    log.info(
        "Профили построены: %d/%d пользователей с позитивами на train",
        int(has_profile.sum()), len(user_ids),
    )

    # Проход 2: косинус для каждого события, пишем инкрементально.
    out_path = features_dir / "emb_cos.parquet"
    schema = pa.schema([(EVENT_ORDER, pa.uint64()), ("emb_cos", pa.float32())])
    writer = pq.ParquetWriter(out_path, schema)
    has_profile_np = has_profile.cpu().numpy()
    try:
        for batch in dataset.to_batches(
            columns=[EVENT_ORDER, c["user_id"], c["item_id"]], batch_size=chunk
        ):
            u = _map_ids(user_ids, batch[c["user_id"]].to_numpy())
            i = _map_ids(item_ids, batch[c["item_id"]].to_numpy())
            mask = (u >= 0) & (i >= 0)
            mask &= np.where(u >= 0, has_profile_np[np.clip(u, 0, None)], False)
            cos = np.full(len(u), np.nan, dtype=np.float32)
            if mask.any():
                u_t = torch.from_numpy(u[mask].astype(np.int64)).to(device)
                i_t = torch.from_numpy(i[mask].astype(np.int64)).to(device)
                cos_t = (profiles[u_t] * emb_t[i_t]).sum(dim=1)
                cos[mask] = cos_t.cpu().numpy()
            order_arr = pa.array(
                batch[EVENT_ORDER].to_numpy().astype(np.uint64), type=pa.uint64()
            )
            writer.write_batch(
                pa.record_batch(
                    [order_arr, pa.array(cos, type=pa.float32())], schema=schema
                )
            )
    finally:
        writer.close()
    log.info("Эмбеддинг-фича записана: %s", out_path)
    return True


def main(cfg: dict) -> None:
    task = init_clearml(cfg, "03_features", task_type="data_processing")
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    interactions_path = processed_dir / "interactions.parquet"
    features_dir = processed_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    bounds = compute_split_bounds(interactions_path, cfg["split"])
    with open(processed_dir / "split.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(bounds, f)
    log.info("Границы сплита по event_order: %s", bounds)

    build_aggregates(cfg, interactions_path, features_dir, bounds["train_end"])

    emb_built = False
    if cfg["features"]["use_embeddings"]:
        emb_built = build_embedding_feature(
            cfg, interactions_path, features_dir, bounds["train_end"]
        )
        if not emb_built:
            log.warning("Эмбеддинг-фича пропущена (нет файлов/колонки эмбеддингов)")

    if task is not None:
        task.get_logger().report_single_value("emb_feature_built", int(emb_built))
        task.upload_artifact("split_bounds", artifact_object=bounds)
    close_clearml(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    setup_logging()
    main(load_config(args.config))
