"""Frame selection for study-level CHD screening."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from agcd.frame_quality import paradigm_a_rank_key


def uniform_subsample(paths: list[str], k: int, *, seed: int, study_id: str) -> list[str]:
    if len(paths) <= k:
        return list(paths)
    rng = random.Random(f"{seed}:{study_id}")
    idx = sorted(rng.sample(range(len(paths)), k))
    return [paths[i] for i in idx]


def select_top_k_by_score(
    frames: list[dict],
    k: int,
    *,
    path_key: str = "path",
    score_key: str = "score",
) -> list[str]:
    """Legacy: rank by precomputed ``score`` (p4c × chamber quality)."""
    ranked = sorted(frames, key=lambda x: float(x.get(score_key, 0.0)), reverse=True)
    return [str(x[path_key]) for x in ranked[:k]]


def _relevance_scalar(frame: dict) -> float:
    key = paradigm_a_rank_key(frame)
    return key[0] * 1000.0 + key[1] * 10.0 + key[2]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def select_top_k_mmr(
    frames: list[dict],
    k: int,
    *,
    relevance_fn: Callable[[dict], float],
    embed_fn: Callable[[dict], np.ndarray | None],
    path_key: str = "path",
    lambda_relevance: float = 0.7,
) -> list[str]:
    """MMR: balance relevance score vs diversity in embedding space."""
    if not frames:
        return []
    k = min(k, len(frames))
    remaining = list(frames)
    selected: list[dict] = []

    while len(selected) < k and remaining:
        if not selected:
            pick = max(remaining, key=relevance_fn)
        else:
            sel_embs = [embed_fn(s) for s in selected]
            sel_embs = [e for e in sel_embs if e is not None]

            def mmr_score(fr: dict) -> float:
                rel = relevance_fn(fr)
                emb = embed_fn(fr)
                if emb is None or not sel_embs:
                    return rel
                max_sim = max(_cosine_sim(emb, se) for se in sel_embs)
                return lambda_relevance * rel - (1.0 - lambda_relevance) * max_sim

            pick = max(remaining, key=mmr_score)
        selected.append(pick)
        remaining.remove(pick)

    return [str(fr[path_key]) for fr in selected]


def select_usable_gate(
    frames: list[dict],
    k: int,
    *,
    path_key: str = "path",
) -> list[str]:
    """Keep all paradigm-A usable frames; optional cap at k (k<=0 → no cap)."""
    usable: list[dict] = []
    for fr in frames:
        paradigm_a_rank_key(fr)
        pa = fr.get("_paradigm_a") or {}
        if pa.get("usable") is not False:
            usable.append(fr)
    pool = usable if usable else list(frames)
    ranked = sorted(pool, key=paradigm_a_rank_key, reverse=True)
    if k <= 0:
        return [str(fr[path_key]) for fr in ranked]
    return [str(fr[path_key]) for fr in ranked[:k]]


def select_top_k_paradigm_a(
    frames: list[dict],
    k: int,
    *,
    path_key: str = "path",
    usable_only: bool = True,
) -> list[str]:
    """Paradigm A: completeness → plane_area → plane_conf (see MASVF spec §4)."""
    scored: list[tuple[tuple[float, float, float], dict]] = []
    for fr in frames:
        key = paradigm_a_rank_key(fr)
        pa = fr.get("_paradigm_a") or {}
        if usable_only and pa.get("usable") is False:
            continue
        scored.append((key, fr))

    if not scored and usable_only:
        for fr in frames:
            scored.append((paradigm_a_rank_key(fr), fr))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [str(fr[path_key]) for _, fr in scored[:k]]


def select_top_k_fetalclip_diverse(
    frames: list[dict],
    k: int,
    embeddings: dict[str, np.ndarray],
    *,
    path_key: str = "path",
    sample_id_key: str = "sample_id",
    lambda_relevance: float = 0.7,
    usable_only: bool = True,
) -> list[str]:
    """Paradigm-A relevance + FetalCLIP MMR diversity within a view bucket."""
    pool = frames
    if usable_only:
        filtered = []
        for fr in frames:
            paradigm_a_rank_key(fr)
            pa = fr.get("_paradigm_a") or {}
            if pa.get("usable") is not False:
                filtered.append(fr)
        if filtered:
            pool = filtered

    def embed_fn(fr: dict) -> np.ndarray | None:
        for key in (fr.get(sample_id_key), fr.get(path_key)):
            if key and key in embeddings:
                return embeddings[key]
        return None

    return select_top_k_mmr(
        pool,
        k,
        relevance_fn=_relevance_scalar,
        embed_fn=embed_fn,
        path_key=path_key,
        lambda_relevance=lambda_relevance,
    )


def select_fetalclip_diverse_gate(
    frames: list[dict],
    k: int,
    embeddings: dict[str, np.ndarray],
    *,
    path_key: str = "path",
    sample_id_key: str = "sample_id",
    lambda_relevance: float = 0.7,
) -> list[str]:
    """Usable-only gate; k<=0 keeps all usable, else MMR top-K on usable pool."""
    usable: list[dict] = []
    for fr in frames:
        paradigm_a_rank_key(fr)
        pa = fr.get("_paradigm_a") or {}
        if pa.get("usable") is not False:
            usable.append(fr)
    pool = usable if usable else list(frames)
    if k <= 0:
        return [str(fr[path_key]) for fr in sorted(pool, key=paradigm_a_rank_key, reverse=True)]
    return select_top_k_fetalclip_diverse(
        pool, k, embeddings,
        path_key=path_key,
        sample_id_key=sample_id_key,
        lambda_relevance=lambda_relevance,
        usable_only=False,
    )


def select_frames(
    frames: list[dict],
    k: int,
    *,
    method: str = "paradigm_a",
    path_key: str = "path",
    score_key: str = "score",
    usable_only: bool = True,
    embeddings: dict[str, np.ndarray] | None = None,
    lambda_relevance: float = 0.7,
) -> list[str]:
    """Unified selector.

    Methods:
      none | legacy | paradigm_a | paradigm_a_gate
      fetalclip_diverse | fetalclip_diverse_gate
    k<=0 for *_gate → keep all usable frames (no top-K cap).
    """
    if method == "none":
        return [str(x[path_key]) for x in frames]
    if method == "legacy":
        return select_top_k_by_score(frames, max(k, 1), path_key=path_key, score_key=score_key)
    if method == "paradigm_a":
        return select_top_k_paradigm_a(frames, max(k, 1), path_key=path_key, usable_only=usable_only)
    if method == "paradigm_a_gate":
        return select_usable_gate(frames, k, path_key=path_key)
    if method == "fetalclip_diverse":
        if not embeddings:
            raise ValueError("fetalclip_diverse requires embeddings dict")
        return select_top_k_fetalclip_diverse(
            frames, max(k, 1), embeddings,
            path_key=path_key,
            lambda_relevance=lambda_relevance,
            usable_only=usable_only,
        )
    if method == "fetalclip_diverse_gate":
        # k<=0: usable gate only — no embeddings needed
        if k <= 0:
            return select_usable_gate(frames, k, path_key=path_key)
        if not embeddings:
            raise ValueError("fetalclip_diverse_gate with k>0 requires embeddings dict")
        return select_fetalclip_diverse_gate(
            frames, k, embeddings,
            path_key=path_key,
            lambda_relevance=lambda_relevance,
        )
    raise ValueError(f"Unknown frame selection method: {method}")
