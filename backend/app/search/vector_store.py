"""FAISS vector indexes.

Three separate indexes, all cosine-similarity (inner product on L2-normalised
vectors):
  - 'clip' : CLIP embeddings for text/image descriptive search
  - 'reid' : OSNet person embeddings for cross-camera re-identification
  - 'face' : InsightFace embeddings for face search

Each index maps a FAISS row directly to a database id (detection_id for clip
and reid, face_id for face) via IndexIDMap2, so a search returns the ids we
join against SQLite. Indexes persist to config.FAISS_DIR.
"""
from __future__ import annotations

import faiss
import numpy as np

from .. import config

_DIMS = {"clip": config.CLIP_DIM, "reid": config.REID_DIM, "face": config.FACE_DIM}
_indexes: dict[str, "faiss.Index"] = {}


def _path(name: str):
    return config.FAISS_DIR / f"{name}.index"


def get_index(name: str):
    """Load (from disk if present) or create the named index."""
    if name not in _indexes:
        p = _path(name)
        if p.exists():
            _indexes[name] = faiss.read_index(str(p))
        else:
            _indexes[name] = faiss.IndexIDMap2(faiss.IndexFlatIP(_DIMS[name]))
    return _indexes[name]


def add(name: str, vectors, ids) -> None:
    """Add vectors [N, dim] with explicit int ids [N]."""
    idx = get_index(name)
    v = np.ascontiguousarray(vectors, dtype="float32")
    if v.ndim == 1:
        v = v[None, :]
    ids = np.ascontiguousarray(np.asarray(ids).ravel(), dtype="int64")
    idx.add_with_ids(v, ids)


def search(name: str, query, top_k: int | None = None):
    """Search one query vector -> (ids, scores) sorted best-first."""
    idx = get_index(name)
    if idx.ntotal == 0:
        return [], []
    q = np.ascontiguousarray(query, dtype="float32")
    if q.ndim == 1:
        q = q[None, :]
    k = min(top_k or config.DEFAULT_TOP_K, idx.ntotal)
    scores, ids = idx.search(q, k)
    return ids[0].tolist(), scores[0].tolist()


def get_vector(name: str, id: int):
    """Reconstruct a stored vector by its db id (detection_id / face_id), or None.

    IndexIDMap2 keeps the original vectors, so we can fetch a detection's own
    embedding without re-running the model - this is what cross-camera tracking
    uses for the reference detection (fast, no model load, no extra VRAM)."""
    idx = get_index(name)
    if idx.ntotal == 0:
        return None
    try:
        return np.asarray(idx.reconstruct(int(id)), dtype="float32")
    except (RuntimeError, TypeError, ValueError):
        return None


def remove(name: str, ids) -> int:
    """Remove vectors by their db ids from the named index (used when a video is
    deleted). Returns how many were removed. Missing ids are ignored."""
    arr = np.ascontiguousarray(np.asarray(list(ids), dtype="int64").ravel())
    if arr.size == 0:
        return 0
    idx = get_index(name)
    try:
        sel = faiss.IDSelectorBatch(arr.size, faiss.swig_ptr(arr))
        return int(idx.remove_ids(sel))
    except Exception:
        return int(idx.remove_ids(arr))


def save(name: str | None = None) -> None:
    for n in ([name] if name else list(_indexes)):
        if n in _indexes:
            faiss.write_index(_indexes[n], str(_path(n)))


def stats() -> dict:
    return {n: get_index(n).ntotal for n in _DIMS}


def reset(name: str | None = None) -> None:
    """Drop index from memory and delete its file (used before re-ingesting)."""
    for n in ([name] if name else list(_DIMS)):
        _indexes.pop(n, None)
        p = _path(n)
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    # Self-test: add, search, persist, reload.
    reset("clip")
    v = np.random.randn(3, config.CLIP_DIM).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    add("clip", v, [11, 22, 33])
    ids, scores = search("clip", v[1], top_k=3)
    print("search ids:", ids, "scores:", [round(s, 3) for s in scores])
    assert ids[0] == 22, f"expected 22 first, got {ids}"
    save("clip")
    _indexes.clear()                      # force reload from disk
    ids2, _ = search("clip", v[1], top_k=1)
    print("after reload, top id:", ids2)
    assert ids2[0] == 22
    print("vector_store OK, stats:", stats())
    reset("clip")
