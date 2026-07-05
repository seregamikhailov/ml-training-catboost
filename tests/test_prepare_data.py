"""Тесты этапа индексации локальных данных (manifest.yaml)."""

from pathlib import Path

import pytest
import yaml

from src.common import load_config
from src.prepare_data import categorize_files, temporal_sort

PATTERNS = load_config("configs/pipeline.yaml")["data_source"]["patterns"]


def test_categorize_files():
    files = [
        "data/raw/interactions/week_00.parquet",
        "data/raw/subsamples/x/train/week_01.parquet",
        "data/raw/metadata/users_metadata.parquet",
        "data/raw/metadata/items_metadata.parquet",
        "data/raw/metadata/item_embeddings.npz",  # 'emb' приоритетнее 'item'
        "data/raw/README.md",                     # не данные -> пропускается
    ]
    cats = categorize_files(files, PATTERNS)
    assert len(cats["interactions"]) == 2
    assert cats["users"] == ["data/raw/metadata/users_metadata.parquet"]
    assert cats["items"] == ["data/raw/metadata/items_metadata.parquet"]
    assert cats["embeddings"] == ["data/raw/metadata/item_embeddings.npz"]


def test_temporal_sort_by_week_number():
    # 'test/week_26' по алфавиту раньше 'train/week_00' — порядок должен
    # быть временной (по номеру недели), а не лексикографический.
    files = [
        "subsamples/x/test/week_26.parquet",
        "subsamples/x/validation/week_25.parquet",
        "subsamples/x/train/week_00.parquet",
        "subsamples/x/train/week_02.parquet",
    ]
    assert temporal_sort(files) == [
        "subsamples/x/train/week_00.parquet",
        "subsamples/x/train/week_02.parquet",
        "subsamples/x/validation/week_25.parquet",
        "subsamples/x/test/week_26.parquet",
    ]


def test_manifest_written_with_temporal_order(pipe_cfg, prepared):
    manifest_path = Path(pipe_cfg["paths"]["raw_dir"]) / "manifest.yaml"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    assert len(manifest["interactions"]) > 0
    weeks = [Path(p).stem for p in manifest["interactions"]]
    assert weeks == sorted(weeks)  # week_00, week_01, ...
    assert len(manifest["users"]) == 1
    assert len(manifest["items"]) == 1
    assert manifest["embeddings"][0].endswith(".npz")


def test_missing_raw_dir_raises(pipe_cfg):
    import copy

    from src import prepare_data

    cfg = copy.deepcopy(pipe_cfg)
    cfg["clearml"]["enabled"] = False
    cfg["paths"]["raw_dir"] = "/nonexistent/raw"
    with pytest.raises(RuntimeError, match="не найден"):
        prepare_data.main(cfg)
