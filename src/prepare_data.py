"""Этап 1. Индексация локальных данных и построение manifest.yaml.

Исходные данные уже лежат в paths.raw_dir (см. README):
    <raw_dir>/**/week_NN.parquet        # недельные шарды взаимодействий
    <raw_dir>/metadata/users_metadata.parquet   # age, gender, geo
    <raw_dir>/metadata/items_metadata.parquet   # duration, author_id
    <raw_dir>/metadata/item_embeddings.npz      # контентные эмбеддинги видео

Этап раскладывает файлы по категориям (interactions / users / items /
embeddings), сортирует шарды взаимодействий по номеру недели (глобальный
временной порядок) и пишет manifest.yaml — его читают следующие этапы.
"""

from __future__ import annotations

import argparse
import re

import yaml

from src.common import (
    close_clearml,
    init_clearml,
    load_config,
    log,
    resolve_path,
    setup_logging,
)

CATEGORY_ORDER = ["embeddings", "interactions", "users", "items"]
DATA_EXTENSIONS = (".parquet", ".npz")

_WEEK_RE = re.compile(r"week_(\d+)")


def temporal_sort(files) -> list[str]:
    """Сортирует шарды по номеру недели (week_NN в имени файла).

    Сортировка по полному пути ломает временной порядок: каталог
    'test/week_26' лексикографически меньше 'train/week_00'."""
    def key(f: str):
        m = _WEEK_RE.search(f)
        return (int(m.group(1)) if m else 10**9, f)

    return sorted(files, key=key)


def categorize_files(files, patterns: dict[str, str]) -> dict[str, list[str]]:
    """Раскладывает файлы по категориям. Категории проверяются в порядке
    CATEGORY_ORDER, файл попадает в первую подошедшую."""
    compiled = {c: re.compile(patterns[c], re.IGNORECASE) for c in CATEGORY_ORDER}
    result: dict[str, list[str]] = {c: [] for c in CATEGORY_ORDER}
    for f in files:
        if not str(f).endswith(DATA_EXTENSIONS):
            continue
        for cat in CATEGORY_ORDER:
            if compiled[cat].search(str(f)):
                result[cat].append(str(f))
                break
    for cat in CATEGORY_ORDER:
        result[cat].sort()
    return result


def main(cfg: dict) -> None:
    task = init_clearml(cfg, "01_prepare_data", task_type="data_processing")
    src_cfg = cfg["data_source"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    if not raw_dir.exists():
        raise RuntimeError(
            f"Каталог с данными не найден: {raw_dir}. "
            "Положите датасет в paths.raw_dir (см. README)."
        )

    # Категоризация — по пути ОТНОСИТЕЛЬНО raw_dir: абсолютный путь может
    # содержать посторонние совпадения (например, /Users/... под macOS).
    rel_files = [
        str(p.relative_to(raw_dir))
        for p in raw_dir.rglob("*")
        if p.suffix in DATA_EXTENSIONS and ".cache" not in p.parts
    ]
    by_cat = categorize_files(rel_files, src_cfg["patterns"])
    by_cat = {
        cat: [str(raw_dir / f) for f in files] for cat, files in by_cat.items()
    }

    interactions = temporal_sort(by_cat["interactions"])
    max_shards = src_cfg.get("max_shards")
    if max_shards:
        interactions = interactions[: int(max_shards)]
        log.info("Ограничение max_shards=%d: берем первые недели", max_shards)

    manifest = {
        "interactions": interactions,
        "users": by_cat["users"],
        "items": by_cat["items"],
        "embeddings": by_cat["embeddings"]
        if cfg["features"]["use_embeddings"]
        else [],
    }
    for cat in ("interactions", "users", "items"):
        if not manifest[cat]:
            raise RuntimeError(
                f"В {raw_dir} не найдено файлов категории '{cat}' "
                f"(паттерн: {src_cfg['patterns'][cat]})"
            )
    log.info(
        "Найдено: %d шардов взаимодействий, %d users, %d items, %d embeddings",
        *(len(manifest[c]) for c in ("interactions", "users", "items", "embeddings")),
    )

    manifest_path = raw_dir / "manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True)
    log.info("Manifest: %s", manifest_path)

    if task is not None:
        task.upload_artifact("manifest", artifact_object=str(manifest_path))
        task.get_logger().report_single_value(
            "interaction_shards", len(manifest["interactions"])
        )
    close_clearml(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    setup_logging()
    main(load_config(args.config))
