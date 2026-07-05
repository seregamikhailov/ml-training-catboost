"""Оффлайн-метрики ранжирования: NDCG@k, MAP@k, MRR@k, HitRate@k,
Recall@k, Precision@k, GAUC.

Все метрики считаются погруппово (группа = пользователь) и усредняются
по группам. Реализация векторизована через numpy (lexsort + reduceat),
поэтому работает на десятках миллионов строк без python-циклов по группам.

Соглашения:
- y_true — градуированная релевантность (label из пайплайна);
- бинарная релевантность для MAP/MRR/HitRate/Recall/GAUC: y_true >= binary_threshold;
- группы без единого позитива исключаются из усреднения бинарных метрик,
  группы с нулевым IDCG — из NDCG (стандартная практика).
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_ranking_metrics", "ndcg_at_k", "map_at_k", "mrr_at_k",
           "hitrate_at_k", "recall_at_k", "precision_at_k", "gauc"]


def _prepare(y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray):
    """Сортирует данные по (группа, -score) и возвращает служебные массивы.

    Возвращает: rel (релевантность в порядке убывания score внутри группы),
    starts (индексы начала групп), sizes, pos_in_group (0-based позиция
    элемента в выдаче своей группы).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    order = np.lexsort((-y_score, group_ids))
    g = group_ids[order]
    rel = y_true[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, len(g)])
    pos_in_group = np.arange(len(g)) - np.repeat(starts, sizes)
    return rel, starts, sizes, pos_in_group


def _group_cumsum(x: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Кумулятивная сумма, обнуляющаяся на границах групп."""
    c = np.cumsum(x)
    offset = np.repeat(c[starts] - x[starts], sizes)
    return c - offset


def ndcg_at_k(y_true, y_score, group_ids, k: int, gain: str = "exp") -> float:
    """NDCG@k с экспоненциальным (2^rel - 1) или линейным гейном."""
    rel, starts, sizes, pos = _prepare(y_true, y_score, group_ids)

    def _gains(r):
        return np.power(2.0, r) - 1.0 if gain == "exp" else r

    discount = 1.0 / np.log2(pos + 2.0)
    in_top = pos < k
    dcg = np.add.reduceat(_gains(rel) * discount * in_top, starts)

    # Идеальный порядок: сортировка по убыванию релевантности внутри группы.
    ideal_rel, i_starts, i_sizes, i_pos = _prepare(y_true, y_true, group_ids)
    i_discount = 1.0 / np.log2(i_pos + 2.0)
    idcg = np.add.reduceat(_gains(ideal_rel) * i_discount * (i_pos < k), i_starts)

    valid = idcg > 0
    if not valid.any():
        return 0.0
    return float(np.mean(dcg[valid] / idcg[valid]))


def map_at_k(y_true, y_score, group_ids, k: int, binary_threshold: float = 1.0) -> float:
    rel, starts, sizes, pos = _prepare(y_true, y_score, group_ids)
    hit = (rel >= binary_threshold).astype(np.float64)
    n_pos = np.add.reduceat(hit, starts)

    cum_hits = _group_cumsum(hit, starts, sizes)
    precision_at_i = cum_hits / (pos + 1.0)
    ap_terms = precision_at_i * hit * (pos < k)
    ap_sum = np.add.reduceat(ap_terms, starts)

    denom = np.minimum(n_pos, k)
    valid = n_pos > 0
    if not valid.any():
        return 0.0
    return float(np.mean(ap_sum[valid] / denom[valid]))


def mrr_at_k(y_true, y_score, group_ids, k: int, binary_threshold: float = 1.0) -> float:
    rel, starts, sizes, pos = _prepare(y_true, y_score, group_ids)
    hit = rel >= binary_threshold
    n_pos = np.add.reduceat(hit.astype(np.float64), starts)

    n = len(rel)
    first_hit = np.where(hit & (pos < k), pos, n)
    min_first = np.minimum.reduceat(first_hit, starts)
    rr = np.where(min_first < n, 1.0 / (min_first + 1.0), 0.0)

    valid = n_pos > 0
    if not valid.any():
        return 0.0
    return float(np.mean(rr[valid]))


def hitrate_at_k(y_true, y_score, group_ids, k: int, binary_threshold: float = 1.0) -> float:
    rel, starts, sizes, pos = _prepare(y_true, y_score, group_ids)
    hit = (rel >= binary_threshold).astype(np.float64)
    n_pos = np.add.reduceat(hit, starts)
    hits_topk = np.add.reduceat(hit * (pos < k), starts)

    valid = n_pos > 0
    if not valid.any():
        return 0.0
    return float(np.mean(hits_topk[valid] > 0))


def recall_at_k(y_true, y_score, group_ids, k: int, binary_threshold: float = 1.0) -> float:
    rel, starts, sizes, pos = _prepare(y_true, y_score, group_ids)
    hit = (rel >= binary_threshold).astype(np.float64)
    n_pos = np.add.reduceat(hit, starts)
    hits_topk = np.add.reduceat(hit * (pos < k), starts)

    valid = n_pos > 0
    if not valid.any():
        return 0.0
    return float(np.mean(hits_topk[valid] / n_pos[valid]))


def precision_at_k(y_true, y_score, group_ids, k: int, binary_threshold: float = 1.0) -> float:
    rel, starts, sizes, pos = _prepare(y_true, y_score, group_ids)
    hit = (rel >= binary_threshold).astype(np.float64)
    n_pos = np.add.reduceat(hit, starts)
    hits_topk = np.add.reduceat(hit * (pos < k), starts)

    valid = n_pos > 0
    if not valid.any():
        return 0.0
    return float(np.mean(hits_topk[valid] / float(k)))


def gauc(y_true, y_score, group_ids, binary_threshold: float = 1.0) -> float:
    """Groupwise AUC: средний по пользователям ROC-AUC (формула Манна-Уитни).

    Учитываются только группы, где есть и позитивы, и негативы.
    Тай-брейк по score — произвольный (без усреднения рангов).
    """
    rel, starts, sizes, pos = _prepare(y_true, y_score, group_ids)
    hit = (rel >= binary_threshold).astype(np.float64)
    n_pos = np.add.reduceat(hit, starts)
    n_neg = sizes.astype(np.float64) - n_pos

    # Ранг по возрастанию score: последний в выдаче группы имеет ранг 1.
    rank_asc = np.repeat(sizes, sizes) - pos
    pos_rank_sum = np.add.reduceat(rank_asc * hit, starts)

    valid = (n_pos > 0) & (n_neg > 0)
    if not valid.any():
        return 0.0
    auc = (pos_rank_sum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0) / (
        n_pos[valid] * n_neg[valid]
    )
    return float(np.mean(auc))


def compute_ranking_metrics(
    y_true,
    y_score,
    group_ids,
    ks: tuple[int, ...] = (5, 10, 20),
    binary_threshold: float = 1.0,
) -> dict[str, float]:
    """Считает полный набор оффлайн-метрик, возвращает плоский dict."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    group_ids = np.asarray(group_ids)

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(y_true, y_score, group_ids, k)
        metrics[f"map@{k}"] = map_at_k(y_true, y_score, group_ids, k, binary_threshold)
        metrics[f"mrr@{k}"] = mrr_at_k(y_true, y_score, group_ids, k, binary_threshold)
        metrics[f"hitrate@{k}"] = hitrate_at_k(y_true, y_score, group_ids, k, binary_threshold)
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_score, group_ids, k, binary_threshold)
        metrics[f"precision@{k}"] = precision_at_k(y_true, y_score, group_ids, k, binary_threshold)
    metrics["gauc"] = gauc(y_true, y_score, group_ids, binary_threshold)
    metrics["n_groups"] = float(len(np.unique(group_ids)))
    return metrics
