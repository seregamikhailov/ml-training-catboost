"""Тесты этапа очистки и построения релевантности (на синтетических данных)."""

import numpy as np
import polars as pl
import pytest

from tests.conftest import RARE_ITEM_ID, RARE_USER_ID


@pytest.fixture(scope="session")
def inter_df(processed) -> pl.DataFrame:
    return pl.read_parquet(processed / "interactions.parquet")


def test_output_exists_and_not_empty(processed, inter_df):
    assert (processed / "interactions.parquet").exists()
    assert (processed / "stats.yaml").exists()
    assert len(inter_df) > 0


def test_anomalous_timespent_filtered(pipe_cfg, inter_df):
    max_ts = pipe_cfg["cleaning"]["max_timespent_sec"]
    assert inter_df["timespent"].max() <= max_ts
    assert inter_df["timespent"].min() >= 0


def test_watch_ratio_clipped(pipe_cfg, inter_df):
    max_wr = pipe_cfg["cleaning"]["max_watch_ratio"]
    assert inter_df["watch_ratio"].max() <= max_wr + 1e-6
    assert inter_df["watch_ratio"].min() >= 0


def test_min_activity_filter_drops_rare(inter_df):
    assert RARE_USER_ID not in inter_df["user_id"].to_list()
    assert RARE_ITEM_ID not in inter_df["item_id"].to_list()


def test_user_and_item_meta_joined(inter_df):
    for col in ("age", "gender", "geo", "duration", "author_id"):
        assert col in inter_df.columns, col
    assert inter_df["duration"].null_count() == 0


def test_event_order_is_monotonic(inter_df):
    order = inter_df["event_order"].to_numpy()
    assert (np.diff(order.astype(np.int64)) > 0).all()


def test_label_formula_matches_config(pipe_cfg, inter_df):
    """Пересчитываем label из сырых колонок по формуле конфига."""
    lbl = pipe_cfg["label"]
    wr = inter_df["watch_ratio"].to_numpy()
    expected = (wr >= lbl["watch_ratio_threshold"]).astype(np.float64) * lbl["watch_weight"]
    for feedback, weight in lbl["weights"].items():
        expected = expected + inter_df[feedback].to_numpy().astype(np.float64) * weight
    expected = np.clip(expected, 0.0, lbl["clip_max"])
    if lbl["dislike_zero"]:
        expected = np.where(inter_df["dislike"].to_numpy() > 0, 0.0, expected)
    np.testing.assert_allclose(inter_df["label"].to_numpy(), expected, atol=1e-5)


def test_dislike_zeroes_label(inter_df):
    disliked = inter_df.filter(pl.col("dislike") > 0)
    assert len(disliked) > 0, "в синтетике должны быть дизлайки"
    assert disliked["label"].max() == 0.0


def test_labels_have_positives(inter_df):
    # Датасет пригоден для ранжирования: есть и позитивы, и негативы.
    share_pos = (inter_df["label"] >= 1.0).mean()
    assert 0.05 < share_pos < 0.95
