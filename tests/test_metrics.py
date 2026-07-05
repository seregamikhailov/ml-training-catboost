"""Тесты оффлайн-метрик: сверка с ручными расчетами и sklearn.

Запуск: python -m pytest tests/ -v
"""

import numpy as np
import pytest

from src.metrics import (
    compute_ranking_metrics,
    gauc,
    hitrate_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ndcg_perfect_ranking():
    # Идеальное ранжирование -> NDCG = 1
    y_true = np.array([3.0, 2.0, 1.0, 0.0])
    y_score = np.array([0.9, 0.7, 0.5, 0.1])
    groups = np.zeros(4, dtype=int)
    assert ndcg_at_k(y_true, y_score, groups, k=4) == pytest.approx(1.0)


def test_ndcg_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 30
        y_true = rng.integers(0, 4, n).astype(float)
        y_score = rng.normal(size=n)
        groups = np.zeros(n, dtype=int)
        if y_true.max() == 0:
            continue
        ours = ndcg_at_k(y_true, y_score, groups, k=10, gain="linear")
        theirs = sklearn_metrics.ndcg_score(y_true[None, :], y_score[None, :], k=10)
        assert ours == pytest.approx(theirs, abs=1e-9)


def test_mrr_hand_computed():
    # Группа 0: первый релевантный на позиции 2 (rr = 1/2)
    # Группа 1: первый релевантный на позиции 1 (rr = 1)
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    y_score = np.array([0.9, 0.8, 0.1, 0.9, 0.5])
    groups = np.array([0, 0, 0, 1, 1])
    assert mrr_at_k(y_true, y_score, groups, k=5) == pytest.approx(0.75)


def test_map_hand_computed():
    # Один запрос: релевантные на позициях 1 и 3.
    # AP@3 = (1/1 + 2/3) / 2 = 5/6
    y_true = np.array([1.0, 0.0, 1.0])
    y_score = np.array([0.9, 0.8, 0.7])
    groups = np.zeros(3, dtype=int)
    assert map_at_k(y_true, y_score, groups, k=3) == pytest.approx(5 / 6)


def test_hitrate_and_recall():
    # Группа 0: 2 позитива, в топ-2 попал 1 -> hit=1, recall=0.5
    # Группа 1: 1 позитив, в топ-2 не попал -> hit=0, recall=0
    y_true = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    y_score = np.array([0.9, 0.8, 0.1, 0.9, 0.8, 0.1])
    groups = np.array([0, 0, 0, 1, 1, 1])
    assert hitrate_at_k(y_true, y_score, groups, k=2) == pytest.approx(0.5)
    assert recall_at_k(y_true, y_score, groups, k=2) == pytest.approx(0.25)
    assert precision_at_k(y_true, y_score, groups, k=2) == pytest.approx(0.25)


def test_gauc_perfect_and_worst():
    y_true = np.array([1.0, 0.0, 0.0, 1.0])
    groups = np.array([0, 0, 1, 1])
    perfect = np.array([0.9, 0.1, 0.1, 0.9])
    worst = np.array([0.1, 0.9, 0.9, 0.1])
    assert gauc(y_true, perfect, groups) == pytest.approx(1.0)
    assert gauc(y_true, worst, groups) == pytest.approx(0.0)


def test_groups_without_positives_are_excluded():
    # Группа 1 без позитивов не должна влиять на бинарные метрики.
    y_true = np.array([1.0, 0.0, 0.0, 0.0])
    y_score = np.array([0.9, 0.1, 0.5, 0.4])
    groups = np.array([0, 0, 1, 1])
    assert mrr_at_k(y_true, y_score, groups, k=2) == pytest.approx(1.0)
    assert hitrate_at_k(y_true, y_score, groups, k=2) == pytest.approx(1.0)


def test_compute_ranking_metrics_keys():
    rng = np.random.default_rng(1)
    n = 1000
    y_true = (rng.random(n) > 0.7).astype(float) * rng.integers(1, 4, n)
    y_score = rng.normal(size=n)
    groups = rng.integers(0, 50, n)
    metrics = compute_ranking_metrics(y_true, y_score, groups, ks=(5, 10))
    expected = {f"{m}@{k}" for m in
                ("ndcg", "map", "mrr", "hitrate", "recall", "precision")
                for k in (5, 10)} | {"gauc", "n_groups"}
    assert set(metrics) == expected
    for name, value in metrics.items():
        if name != "n_groups":
            assert 0.0 <= value <= 1.0, name
