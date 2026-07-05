"""Этап 5. Обучение CatBoost-ранкера (YetiRank) на GPU + оффлайн-оценка.

- группы = user_id (объекты одной группы идут подряд — это гарантирует
  build_dataset);
- метрики на этапе обучения: eval_metric/custom_metric из configs/train.yaml,
  CatBoost считает их на валидации по итерациям, кривые логируются в ClearML;
- после обучения — полный набор наших оффлайн-метрик (src/metrics.py) на
  val и test;
- артефакты: модель .cbm, feature importance, метрики.

Если CUDA недоступна (локальный Mac) и allow_cpu_fallback=true — обучение
автоматически падает обратно на CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.common import (
    close_clearml,
    init_clearml,
    load_config,
    log,
    resolve_path,
    set_seed,
    setup_logging,
)
from src.metrics import compute_ranking_metrics


def load_split(dataset_dir: Path, split: str, meta: dict, max_rows: int | None = None):
    """Возвращает (X: pandas.DataFrame, y, group_id) для сплита.

    max_rows — ограничение для быстрых проверочных прогонов: берем первые
    N строк и отбрасываем последнюю (возможно, обрезанную) группу, чтобы
    не подавать CatBoost неполного пользователя."""
    if max_rows:
        df = pl.scan_parquet(dataset_dir / f"{split}.parquet").head(max_rows).collect()
        if df[meta["group"]].n_unique() > 1:
            last_group = df[meta["group"]][-1]
            df = df.filter(pl.col(meta["group"]) != last_group)
        log.info("[%s] ограничение max_rows=%d -> %d строк", split, max_rows, len(df))
    else:
        df = pl.read_parquet(dataset_dir / f"{split}.parquet")
    features = meta["num_features"] + meta["cat_features"]
    X = df.select(features).to_pandas()
    for col in meta["cat_features"]:
        X[col] = X[col].astype(str)
    y = df[meta["label"]].to_numpy()
    group_id = df[meta["group"]].to_numpy()
    return X, y, group_id


def make_pool(X, y, group_id, meta: dict):
    from catboost import Pool

    return Pool(
        data=X,
        label=y,
        group_id=group_id,
        cat_features=meta["cat_features"],
    )


def fit_model(model_cfg: dict, train_pool, val_pool):
    """Обучает CatBoostRanker; при отсутствии GPU опционально падает на CPU."""
    from catboost import CatBoostError, CatBoostRanker

    params = dict(model_cfg)
    allow_fallback = params.pop("allow_cpu_fallback", True)

    try:
        model = CatBoostRanker(**params)
        model.fit(train_pool, eval_set=val_pool)
        return model, params["task_type"]
    except CatBoostError as e:
        if params.get("task_type") != "GPU" or not allow_fallback:
            raise
        log.warning("GPU-обучение не удалось (%s), переходим на CPU", e)
        params["task_type"] = "CPU"
        params.pop("devices", None)
        model = CatBoostRanker(**params)
        model.fit(train_pool, eval_set=val_pool)
        return model, "CPU"


def report_training_curves(task, model) -> None:
    """Кривые обучения (метрики по итерациям) -> ClearML scalars."""
    if task is None:
        return
    logger = task.get_logger()
    evals = model.get_evals_result()
    for split_name, metrics in evals.items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values):
                logger.report_scalar(
                    title=metric_name,
                    series=split_name,
                    value=float(value),
                    iteration=iteration,
                )


def evaluate_split(model, X, y, group_id, eval_cfg: dict) -> dict[str, float]:
    scores = model.predict(X)
    return compute_ranking_metrics(
        y_true=y,
        y_score=scores,
        group_ids=group_id,
        ks=tuple(eval_cfg["ks"]),
        binary_threshold=eval_cfg["binary_threshold"],
    )


def main(cfg: dict) -> None:
    set_seed(int(cfg["model"].get("random_seed", 42)))
    task = init_clearml(cfg, cfg["clearml"].get("task_name", "catboost_ranker"),
                        task_type="training")
    if task is not None:
        task.connect(cfg["model"], name="model")

    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"])
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "dataset_meta.yaml", "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    log.info(
        "Фичи: %d числовых + %d категориальных; размеры: %s",
        len(meta["num_features"]), len(meta["cat_features"]), meta["sizes"],
    )

    max_rows = cfg.get("data", {}).get("max_rows")
    X_train, y_train, g_train = load_split(dataset_dir, "train", meta, max_rows)
    X_val, y_val, g_val = load_split(dataset_dir, "val", meta, max_rows)
    train_pool = make_pool(X_train, y_train, g_train, meta)
    val_pool = make_pool(X_val, y_val, g_val, meta)

    model, used_device = fit_model(cfg["model"], train_pool, val_pool)
    log.info(
        "Обучение завершено на %s, best_iteration=%s, best_score=%s",
        used_device, model.get_best_iteration(), model.get_best_score(),
    )
    report_training_curves(task, model)

    # --- финальная оффлайн-оценка на val и test ---------------------------
    results = {"val": evaluate_split(model, X_val, y_val, g_val, cfg["eval"])}
    del X_train, y_train, train_pool  # память под test

    X_test, y_test, g_test = load_split(dataset_dir, "test", meta, max_rows)
    results["test"] = evaluate_split(model, X_test, y_test, g_test, cfg["eval"])

    for split, metrics in results.items():
        log.info("Метрики [%s]:", split)
        for name, value in metrics.items():
            log.info("  %-14s %.5f", name, value)

    # --- артефакты ----------------------------------------------------------
    model_path = models_dir / "catboost_ranker.cbm"
    model.save_model(str(model_path))

    fi = model.get_feature_importance(val_pool, prettified=True)
    fi_path = models_dir / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    log.info("Топ-10 фичей:\n%s", fi.head(10).to_string(index=False))

    metrics_path = models_dir / "offline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if task is not None:
        logger = task.get_logger()
        for split, metrics in results.items():
            for name, value in metrics.items():
                logger.report_single_value(f"{split}/{name}", round(float(value), 6))
        logger.report_table(
            title="feature_importance", series=used_device, iteration=0, table_plot=fi
        )
        task.upload_artifact("model", artifact_object=str(model_path))
        task.upload_artifact("offline_metrics", artifact_object=results)
        task.upload_artifact("feature_importance", artifact_object=str(fi_path))
    close_clearml(task)

    log.info("Готово. Модель: %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    setup_logging()
    main(load_config(args.config))
