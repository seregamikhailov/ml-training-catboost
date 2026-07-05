"""Этап 4. Сборка train/val/test датасетов для CatBoost-ранкера.

- джойним агрегатные и эмбеддинг-фичи к событиям;
- темпоральный сплит по границам из split.yaml (train -> val -> test);
- сортируем каждый сплит по (user_id, event_order): CatBoost требует,
  чтобы объекты одной группы шли подряд;
- пишем parquet + dataset_meta.yaml со списками фичей (его читает train.py).

Важно про утечки: фидбек текущего события (like, timespent, watch_ratio...)
в фичи НЕ входит — это сигнал таргета. Контекст показа (place, platform,
agent) известен на момент ранжирования, поэтому используется.
"""

from __future__ import annotations

import argparse

import polars as pl
import yaml

from src.common import (
    close_clearml,
    init_clearml,
    load_config,
    log,
    resolve_path,
    setup_logging,
)
from src.preprocess import EVENT_ORDER, LABEL

SPLITS = ("train", "val", "test")


def main(cfg: dict) -> None:
    task = init_clearml(cfg, "04_build_dataset", task_type="data_processing")
    c = cfg["columns"]
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    features_dir = processed_dir / "features"
    dataset_dir = processed_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "split.yaml", "r", encoding="utf-8") as f:
        bounds = yaml.safe_load(f)

    lf = pl.scan_parquet(processed_dir / "interactions.parquet")

    user_feats = pl.scan_parquet(features_dir / "user_features.parquet")
    item_feats = pl.scan_parquet(features_dir / "item_features.parquet")
    author_feats = pl.scan_parquet(features_dir / "author_features.parquet")
    ua_feats = pl.scan_parquet(features_dir / "ua_features.parquet")

    lf = (
        lf.join(user_feats, on=c["user_id"], how="left")
        .join(item_feats, on=c["item_id"], how="left")
        .join(author_feats, on=c["author_id"], how="left")
        .join(ua_feats, on=[c["user_id"], c["author_id"]], how="left")
    )

    emb_path = features_dir / "emb_cos.parquet"
    has_emb = emb_path.exists()
    if has_emb:
        emb = pl.scan_parquet(emb_path).with_columns(
            pl.col(EVENT_ORDER).cast(pl.UInt32)
        )
        lf = lf.join(emb, on=EVENT_ORDER, how="left")

    # user-item фича: насколько длительность видео похожа на привычную юзеру
    lf = lf.with_columns(
        (pl.col(c["duration"]) - pl.col("user_mean_duration"))
        .abs()
        .cast(pl.Float32)
        .alias("dur_diff")
    )

    schema_cols = set(lf.collect_schema().names())
    feedback = list(c["feedback"])

    # --- списки фичей -------------------------------------------------------
    num_features = [
        col
        for col in (
            # юзерские
            "age",
            "user_n_events", "user_mean_watch_ratio", "user_mean_timespent",
            "user_mean_duration", "user_n_authors",
            *[f"user_{f}_rate" for f in feedback],
            # айтемные
            c["duration"],
            "item_n_views", "item_mean_watch_ratio", "item_mean_label",
            *[f"item_{f}_rate" for f in feedback],
            "author_n_views", "author_n_items", "author_mean_watch_ratio",
            *[f"author_{f}_rate" for f in feedback],
            # юзер-айтемные
            "ua_n_events", "ua_mean_watch_ratio", "ua_mean_label",
            "dur_diff",
            *(["emb_cos"] if has_emb else []),
        )
        if col in schema_cols or col == "dur_diff"
    ]
    cat_features = [
        col
        for col in ("gender", "geo", *c["context"])
        if col in schema_cols
    ]

    lf = lf.with_columns(
        # strict=False: если, например, age приедет строкой-бакетом,
        # некастуемые значения станут null (CatBoost умеет в NaN).
        [pl.col(col).cast(pl.Float32, strict=False) for col in num_features]
        + [
            pl.col(col).cast(pl.Utf8).fill_null("unknown").alias(col)
            for col in cat_features
        ]
    )

    keep = [c["user_id"], c["item_id"], EVENT_ORDER, LABEL, *num_features, *cat_features]
    lf = lf.select(keep)

    split_expr = {
        "train": pl.col(EVENT_ORDER) < bounds["train_end"],
        "val": (pl.col(EVENT_ORDER) >= bounds["train_end"])
        & (pl.col(EVENT_ORDER) < bounds["val_end"]),
        "test": pl.col(EVENT_ORDER) >= bounds["val_end"],
    }

    sizes = {}
    for split in SPLITS:
        out = dataset_dir / f"{split}.parquet"
        log.info("Пишем %s ...", out)
        (
            lf.filter(split_expr[split])
            .sort([c["user_id"], EVENT_ORDER])
            .sink_parquet(out)
        )
        sizes[split] = int(
            pl.scan_parquet(out).select(pl.len()).collect().item()
        )
    log.info("Размеры сплитов: %s", sizes)

    meta = {
        "label": LABEL,
        "group": c["user_id"],
        "item_id": c["item_id"],
        "event_order": EVENT_ORDER,
        "num_features": num_features,
        "cat_features": cat_features,
        "sizes": sizes,
        "split_bounds": bounds,
    }
    meta_path = dataset_dir / "dataset_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    log.info(
        "Датасет собран: %d числовых, %d категориальных фичей",
        len(num_features), len(cat_features),
    )

    if task is not None:
        logger = task.get_logger()
        for split, n in sizes.items():
            logger.report_single_value(f"rows_{split}", n)
        task.upload_artifact("dataset_meta", artifact_object=meta)
    close_clearml(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    setup_logging()
    main(load_config(args.config))
