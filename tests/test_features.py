"""Тесты этапа фичей: агрегаты по train-окну, отсутствие утечки будущего,
эмбеддинг-фича."""

import numpy as np
import polars as pl
import pytest
import yaml

from src.features import _map_ids


@pytest.fixture(scope="session")
def split_bounds(featured) -> dict:
    with open(featured / "split.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def train_window(featured, split_bounds) -> pl.DataFrame:
    return pl.read_parquet(featured / "interactions.parquet").filter(
        pl.col("event_order") < split_bounds["train_end"]
    )


def test_all_feature_files_written(featured):
    fdir = featured / "features"
    for name in ("user_features", "item_features", "author_features",
                 "ua_features", "emb_cos"):
        assert (fdir / f"{name}.parquet").exists(), name


def test_user_aggregates_match_recomputation(featured, train_window):
    """user_n_events и user_mean_watch_ratio сходятся с пересчетом руками."""
    got = pl.read_parquet(featured / "features" / "user_features.parquet")
    expected = train_window.group_by("user_id").agg(
        pl.len().alias("n"), pl.col("watch_ratio").mean().alias("wr")
    )
    joined = got.join(expected, on="user_id", how="inner")
    assert len(joined) == len(got) == len(expected)
    assert (joined["user_n_events"] == joined["n"]).all()
    np.testing.assert_allclose(
        joined["user_mean_watch_ratio"].to_numpy(), joined["wr"].to_numpy(), atol=1e-5
    )


def test_item_aggregates_use_train_window_only(featured, train_window):
    """Просмотры из val/test не должны попадать в item_n_views (утечка будущего)."""
    got = pl.read_parquet(featured / "features" / "item_features.parquet")
    expected = train_window.group_by("item_id").len()
    joined = got.join(expected, on="item_id", how="inner")
    assert (joined["item_n_views"] == joined["len"]).all()
    # Айтемы, встречающиеся только после train_end, отсутствуют в фичах.
    assert set(got["item_id"].to_list()) == set(expected["item_id"].to_list())


def test_ua_aggregates_match_recomputation(featured, train_window):
    got = pl.read_parquet(featured / "features" / "ua_features.parquet")
    expected = train_window.group_by(["user_id", "author_id"]).len()
    joined = got.join(expected, on=["user_id", "author_id"], how="inner")
    assert len(joined) == len(got) == len(expected)
    assert (joined["ua_n_events"] == joined["len"]).all()


def test_emb_cos_values(featured):
    emb = pl.read_parquet(featured / "features" / "emb_cos.parquet")
    inter_len = len(pl.read_parquet(featured / "interactions.parquet"))
    assert len(emb) == inter_len  # по строке на событие
    values = emb["emb_cos"].drop_nulls().drop_nans().to_numpy()
    assert len(values) > 0
    assert values.min() >= -1.0 - 1e-5
    assert values.max() <= 1.0 + 1e-5


def test_map_ids():
    sorted_ids = np.array([10, 20, 30, 40])
    query = np.array([30, 10, 99, 40, 5])
    np.testing.assert_array_equal(
        _map_ids(sorted_ids, query), np.array([2, 0, -1, 3, -1])
    )
