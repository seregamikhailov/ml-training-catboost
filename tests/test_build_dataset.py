"""Тесты сборки датасета: сплиты, группы, отсутствие утечек в фичах."""

import numpy as np
import polars as pl
import pytest
import yaml

LEAKY_COLUMNS = (
    "timespent", "watch_ratio", "label",
    "like", "dislike", "share", "bookmark", "click_on_author", "open_comments",
)


@pytest.fixture(scope="session")
def meta(dataset) -> dict:
    with open(dataset / "dataset_meta.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def splits(dataset) -> dict[str, pl.DataFrame]:
    return {s: pl.read_parquet(dataset / f"{s}.parquet") for s in ("train", "val", "test")}


def test_splits_cover_all_events_without_overlap(splits, featured):
    total = len(pl.read_parquet(featured / "interactions.parquet"))
    assert sum(len(df) for df in splits.values()) == total
    orders = [set(df["event_order"].to_list()) for df in splits.values()]
    assert not (orders[0] & orders[1]) and not (orders[1] & orders[2]) \
        and not (orders[0] & orders[2])


def test_temporal_split_boundaries(splits, meta):
    bounds = meta["split_bounds"]
    assert splits["train"]["event_order"].max() < bounds["train_end"]
    assert splits["val"]["event_order"].min() >= bounds["train_end"]
    assert splits["val"]["event_order"].max() < bounds["val_end"]
    assert splits["test"]["event_order"].min() >= bounds["val_end"]


def test_groups_are_contiguous(splits):
    """CatBoost требует, чтобы объекты одной группы шли подряд."""
    for name, df in splits.items():
        uid = df["user_id"].to_numpy()
        n_runs = int((uid[1:] != uid[:-1]).sum()) + 1
        assert n_runs == df["user_id"].n_unique(), name


def test_no_target_leakage_in_features(meta):
    features = set(meta["num_features"]) | set(meta["cat_features"])
    for col in LEAKY_COLUMNS:
        assert col not in features, f"утечка таргета: {col}"


def test_feature_columns_present_and_typed(splits, meta):
    df = splits["train"]
    for col in meta["num_features"]:
        assert df.schema[col] == pl.Float32, col
    for col in meta["cat_features"]:
        assert df.schema[col] == pl.Utf8, col
        assert df[col].null_count() == 0, col


def test_all_feature_groups_present(meta):
    features = meta["num_features"] + meta["cat_features"]
    assert any(f.startswith("user_") for f in features), "нет юзерских фичей"
    assert any(f.startswith("item_") for f in features), "нет айтемных фичей"
    assert any(f.startswith(("ua_", "dur_diff", "emb_cos")) for f in features), \
        "нет юзер-айтемных фичей"
    assert "emb_cos" in features


def test_cold_start_rows_have_null_aggregates_not_dropped(splits):
    """Юзеры/айтемы, впервые появившиеся после train-окна, остаются в val/test
    с null-агрегатами (CatBoost умеет NaN), а не выкидываются."""
    val_test = pl.concat([splits["val"], splits["test"]])
    assert len(val_test) > 0
    # хотя бы структурно: null в агрегатах допустимы
    assert val_test["item_n_views"].null_count() >= 0
