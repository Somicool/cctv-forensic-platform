"""Appearance-assisted association guard for person tracking.

The problem
-----------
ByteTrack associates purely on MOTION (IoU + a Kalman prediction). When two
people pass each other, or one is occluded and another walks into the same place,
motion says "same object" and the track id silently transfers to the wrong
person. Downstream everything - the tracking box, the cross-camera search, the
journey - then follows the wrong human.

The rule this module enforces
-----------------------------
Motion may PROPOSE an association. Appearance DECIDES it.

Every time ByteTrack claims a detection belongs to an existing track, the crop's
person ReID embedding is compared against that track's stored appearance. If it
does not look like the same person, the association is REFUSED. The detection is
then either re-acquired onto a track that was recently lost (the person who was
briefly occluded) or, failing that, given a fresh identity. The original track is
left untouched, so a wrong association can never overwrite a good identity.

Track state
-----------
Each live track keeps up to config.TRACK_MAX_EMBEDDINGS REPRESENTATIVE embeddings
rather than one crop (Part 6). They are chosen for diversity: a new view replaces
the one most similar to it, so the set spans poses, distances and lighting
instead of collecting near-duplicates of the same frame. Matching is set-to-set,
which is what makes recognition survive a posture change.

Tracks unseen for a while move to a LOST pool and stay eligible for
re-acquisition for config.TRACK_LOST_WINDOW sampled frames, so a brief occlusion
does not cost the person their identity.

This module never touches the detector, the database or the search indexes. It
only rewrites the track_id on detections as they stream out of the tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config


def _unit(v):
    v = np.asarray(v, dtype="float32").ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


@dataclass
class TrackAppearance:
    """Appearance memory of one person track."""
    track_id: int
    embeddings: list = field(default_factory=list)   # representative views
    hits: int = 0                                    # times confirmed
    first_frame: int = 0
    last_frame: int = 0
    rejected: int = 0                                # associations refused
    last_emb: object = None                          # most recently confirmed view

    def recent_similarity(self, emb) -> float:
        """Similarity to the LAST confirmed view.

        An identity handover shows up here first: the historical view set can still
        contain a good match for both people, but the frame-to-frame appearance
        breaks the instant the box jumps to someone else."""
        if self.last_emb is None:
            return 1.0
        return max(0.0, min(1.0, float(np.dot(emb, self.last_emb))))

    def similarity(self, emb) -> float:
        """Set-to-set similarity: best matching stored view, softened by the mean.

        The best-view term is what lets a seated or turned person still match; the
        mean term stops one lucky near-duplicate from carrying a bad match."""
        if not self.embeddings:
            return 1.0                               # nothing to contradict yet
        sims = np.array([float(np.dot(emb, e)) for e in self.embeddings], dtype="float32")
        best = float(sims.max())
        mean = float(sims.mean())
        return max(0.0, min(1.0, 0.7 * best + 0.3 * mean))

    def add(self, emb, frame: int) -> None:
        """Keep a DIVERSE set of views, not the most recent ones."""
        self.hits += 1
        self.last_frame = frame
        self.last_emb = emb
        if not self.embeddings:
            self.embeddings.append(emb)
            self.first_frame = frame
            return
        if len(self.embeddings) < config.TRACK_MAX_EMBEDDINGS:
            self.embeddings.append(emb)
            return
        # replace the stored view this one is most redundant with, so the set
        # keeps spanning different poses/distances instead of one moment
        sims = [float(np.dot(emb, e)) for e in self.embeddings]
        i = int(np.argmax(sims))
        if sims[i] > 0.97:                           # near-duplicate: swap it out
            self.embeddings[i] = emb


class IdentityGuard:
    """Verifies ByteTrack associations by appearance and re-acquires lost tracks."""

    def __init__(self, min_sim: float | None = None, reacquire_sim: float | None = None,
                 lost_window: int | None = None, warmup: int | None = None,
                 recent_min: float | None = None, id_offset: int = 100_000):
        self.min_sim = config.TRACK_REID_MIN_SIM if min_sim is None else min_sim
        self.recent_min = (config.TRACK_REID_RECENT_MIN if recent_min is None else recent_min)
        self.reacquire_sim = (config.TRACK_REACQUIRE_MIN_SIM if reacquire_sim is None
                              else reacquire_sim)
        self.lost_window = config.TRACK_LOST_WINDOW if lost_window is None else lost_window
        self.warmup = config.TRACK_GUARD_WARMUP if warmup is None else warmup
        self.live: dict[int, TrackAppearance] = {}
        self.lost: dict[int, TrackAppearance] = {}
        # ids we mint ourselves start high so they can never collide with
        # ByteTrack's own numbering
        self._next_id = id_offset
        self.stats = {"checked": 0, "accepted": 0, "rejected": 0,
                      "reacquired": 0, "new_identities": 0, "switches_blocked": 0}
        # maps a ByteTrack id to the identity we decided it currently represents
        self._alias: dict[int, int] = {}

    # ---------------------------------------------------------------- helpers
    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _best_lost(self, emb, frame: int):
        """Most similar recently-lost track, if any clears the re-acquire bar."""
        best, best_sim = None, 0.0
        for t in self.lost.values():
            if frame - t.last_frame > self.lost_window:
                continue
            s = t.similarity(emb)
            if s > best_sim:
                best, best_sim = t, s
        return (best, best_sim) if (best and best_sim >= self.reacquire_sim) else (None, best_sim)

    def retire(self, frame: int, seen_ids: set) -> None:
        """Move unseen live tracks to the lost pool; forget the truly stale.

        Every live track that was not confirmed this frame becomes LOST, never
        discarded: its appearance memory is exactly what re-acquisition needs a
        moment later. Memory is only released once the track has been gone longer
        than the configured window."""
        for tid in [t for t in self.live if t not in seen_ids]:
            self.lost[tid] = self.live.pop(tid)
        for tid in [t for t, v in self.lost.items() if frame - v.last_frame > self.lost_window]:
            self.lost.pop(tid, None)
            # its alias must go too, or a recycled ByteTrack id would inherit it
            for bt, ident in list(self._alias.items()):
                if ident == tid:
                    self._alias.pop(bt, None)

    # ---------------------------------------------------------------- main
    def resolve(self, bt_id: int, emb, frame: int) -> tuple[int, str]:
        """Decide which identity this detection really belongs to.

        Returns (identity_id, action) where action is one of
        `accept` / `reacquire` / `reject-new` / `new`."""
        emb = _unit(emb)
        self.stats["checked"] += 1
        ident = self._alias.get(bt_id, bt_id)
        track = self.live.get(ident) or self.lost.get(ident)

        # --- brand-new ByteTrack id: is this someone we just lost? ----------
        if track is None:
            cand, _sim = self._best_lost(emb, frame)
            if cand is not None:
                self.lost.pop(cand.track_id, None)
                self.live[cand.track_id] = cand
                cand.add(emb, frame)
                self._alias[bt_id] = cand.track_id
                self.stats["reacquired"] += 1
                return cand.track_id, "reacquire"
            t = TrackAppearance(track_id=ident, first_frame=frame, last_frame=frame)
            t.add(emb, frame)
            self.live[ident] = t
            self._alias[bt_id] = ident
            self.stats["new_identities"] += 1
            return ident, "new"

        # --- existing track: does the appearance agree? ----------------------
        # BOTH tests must pass: the track's accumulated views (tolerant of posture
        # and lighting change over time) and the immediately preceding view (which
        # is where a hand-over to another person shows up first).
        sim = track.similarity(emb)
        recent = track.recent_similarity(emb)
        if track.hits < self.warmup or (sim >= self.min_sim and recent >= self.recent_min):
            self.lost.pop(ident, None)
            self.live[ident] = track
            track.add(emb, frame)
            self._alias[bt_id] = ident
            self.stats["accepted"] += 1
            return ident, "accept"

        # Appearance says this is NOT the same person. Refuse the association.
        track.rejected += 1
        self.stats["rejected"] += 1
        self.stats["switches_blocked"] += 1
        cand, _sim = self._best_lost(emb, frame)
        if cand is not None and cand.track_id != ident:
            self.lost.pop(cand.track_id, None)
            self.live[cand.track_id] = cand
            cand.add(emb, frame)
            self._alias[bt_id] = cand.track_id
            self.stats["reacquired"] += 1
            return cand.track_id, "reacquire"
        new_id = self._new_id()
        t = TrackAppearance(track_id=new_id, first_frame=frame, last_frame=frame)
        t.add(emb, frame)
        self.live[new_id] = t
        self._alias[bt_id] = new_id
        self.stats["new_identities"] += 1
        return new_id, "reject-new"

    def summary(self) -> dict:
        s = dict(self.stats)
        s["live_tracks"] = len(self.live)
        s["lost_tracks"] = len(self.lost)
        total = max(1, s["checked"])
        s["reject_rate"] = round(s["rejected"] / total, 4)
        return s
