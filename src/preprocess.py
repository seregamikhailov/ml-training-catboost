"""Этап 2. Очистка данных и построение градуированной релевантности.

Вход: сырые parquet из data/raw (по manifest.yaml).
Выход: data/processed/interactions.parquet — очищенные интеракции с
колонками event_order (глобальный временной порядок), watch_ratio, label
и присоединенными мета-данными user/item.

Вся обработка — на polars в lazy/streaming режиме, чтобы не упираться
в память при больших объемах.
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

EVENT_ORDER = "event_order"
WATCH_RATIO = "watch_ratio"
LABEL = "label"


def load_manifest(cfg: dict) -> dict:
    manifest_path = resolve_path(cfg["paths"]["raw_dir"]) / "manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_label_expr(cfg: dict) -> pl.Expr:
    """label = watch_weight * [watch_ratio >= thr] + sum(w_f * feedback_f);
    дизлайк (опционально) обнуляет релевантность; сверху клип."""
    lbl = cfg["label"]
    expr = (
        (pl.col(WATCH_RATIO) >= lbl["watch_ratio_threshold"]).cast(pl.Float32)
        * lbl["watch_weight"]
    )
    for feedback, weight in lbl["weights"].items():
        expr = expr + pl.col(feedback).cast(pl.Float32) * weight
    expr = expr.clip(0.0, lbl["clip_max"])
    if lbl.get("dislike_zero", True) and "dislike" in cfg["columns"]["feedback"]:
        expr = pl.when(pl.col("dislike") > 0).then(0.0).otherwise(expr)
    return expr.cast(pl.Float32).alias(LABEL)


def scan_interactions(cfg: dict, manifest: dict) -> pl.LazyFrame:
    from src.prepare_data import temporal_sort

    c = cfg["columns"]
    files = temporal_sort(manifest["interactions"])
    if not files:
        raise RuntimeError("В manifest.yaml нет шардов взаимодействий — сначала запустите этап prepare")
    lf = pl.scan_parquet(files)
    # Глобальный временной порядок: датасет отсортирован по времени,
    # порядок файлов и строк в них его сохраняет.
    lf = lf.with_row_index(EVENT_ORDER)

    feedback = [f for f in c["feedback"]]
    lf = lf.with_columns([pl.col(f).cast(pl.Int8).fill_null(0) for f in feedback])
    lf = lf.drop_nulls([c["user_id"], c["item_id"]])
    return lf


def scan_meta(files: list[str], keep_cols: list[str]) -> pl.LazyFrame | None:
    if not files:
        return None
    lf = pl.scan_parquet(sorted(files))
    schema_cols = lf.collect_schema().names()
    cols = [col for col in keep_cols if col in schema_cols]
    return lf.select(cols)


def main(cfg: dict) -> None:
    task = init_clearml(cfg, "02_preprocess", task_type="data_processing")
    c = cfg["columns"]
    clean = cfg["cleaning"]
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg)

    lf = scan_interactions(cfg, manifest)

    # --- метаданные видео: duration, author_id (эмбеддинги здесь не нужны)
    items = scan_meta(manifest["items"], [c["item_id"], c["duration"], c["author_id"]])
    if items is not None:
        lf = lf.join(items, on=c["item_id"], how="left")
    elif c["duration"] not in lf.collect_schema().names():
        raise RuntimeError("Нет ни item-метаданных, ни колонки duration в интеракциях")

    # --- метаданные пользователей: age, gender, geo
    users = scan_meta(manifest["users"], [c["user_id"]] + list(c["user_static"]))
    if users is not None:
        lf = lf.join(users, on=c["user_id"], how="left")

    # --- опциональное прореживание по пользователям (детерминированный хэш)
    sample = cfg.get("sample", {})
    if sample.get("user_fraction"):
        frac = float(sample["user_fraction"])
        lf = lf.filter(
            pl.col(c["user_id"]).hash(seed=sample.get("seed", 42)) % 10_000
            < int(frac * 10_000)
        )
        log.info("Прореживание пользователей: доля %.3f", frac)

    # --- очистка
    lf = lf.filter(
        pl.col(c["timespent"]).is_not_null()
        & (pl.col(c["timespent"]) >= 0)
        & (pl.col(c["timespent"]) <= clean["max_timespent_sec"])
        & pl.col(c["duration"]).is_not_null()
        & (pl.col(c["duration"]) > 0)
    )
    lf = lf.with_columns(
        (pl.col(c["timespent"]) / pl.col(c["duration"]))
        .clip(0.0, clean["max_watch_ratio"])
        .cast(pl.Float32)
        .alias(WATCH_RATIO)
    )

    # --- фильтр по минимальной активности (semi-join на агрегаты)
    active_users = (
        lf.group_by(c["user_id"])
        .len()
        .filter(pl.col("len") >= clean["min_user_interactions"])
        .select(c["user_id"])
    )
    active_items = (
        lf.group_by(c["item_id"])
        .len()
        .filter(pl.col("len") >= clean["min_item_interactions"])
        .select(c["item_id"])
    )
    lf = lf.join(active_users, on=c["user_id"], how="semi")
    lf = lf.join(active_items, on=c["item_id"], how="semi")

    # --- градуированная релевантность
    lf = lf.with_columns(build_label_expr(cfg))
    lf = lf.sort(EVENT_ORDER)

    out_path = processed_dir / "interactions.parquet"
    log.info("Пишем %s (streaming)...", out_path)
    lf.sink_parquet(out_path)

    # --- статистика этапа
    stats_lf = pl.scan_parquet(out_path).select(
        n_rows=pl.len(),
        n_users=pl.col(c["user_id"]).n_unique(),
        n_items=pl.col(c["item_id"]).n_unique(),
        label_mean=pl.col(LABEL).mean(),
        label_pos_share=(pl.col(LABEL) >= 1.0).mean(),
        watch_ratio_mean=pl.col(WATCH_RATIO).mean(),
    )
    stats = {k: float(v) for k, v in stats_lf.collect().to_dicts()[0].items()}
    log.info("Статистика после очистки: %s", stats)

    with open(processed_dir / "stats.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f)

    if task is not None:
        logger = task.get_logger()
        for name, value in stats.items():
            logger.report_single_value(name, value)
        task.upload_artifact("stats", artifact_object=stats)
    close_clearml(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    setup_logging()
    main(load_config(args.config))
