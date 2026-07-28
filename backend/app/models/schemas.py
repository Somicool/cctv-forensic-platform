"""Pydantic models = the API contract.

This file is the single source of truth for request/response shapes so the
frontend can be built in parallel against these exact structures.
"""
from typing import Optional
from pydantic import BaseModel, Field


# ----------------------------- Cameras -----------------------------
class Camera(BaseModel):
    camera_id: str
    name: Optional[str] = None
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


# ----------------------------- Filters -----------------------------
class SearchFilters(BaseModel):
    cameras: Optional[list[str]] = None          # camera_ids to include
    video_id: Optional[int] = None               # restrict search to ONE recording
    start_time: Optional[str] = None             # ISO datetime
    end_time: Optional[str] = None               # ISO datetime
    object_type: Optional[str] = None            # "person" | "vehicle" | None
    colors: Optional[list[str]] = None
    vehicle_type: Optional[str] = None
    min_confidence: float = 0.0


# ----------------------------- Search ------------------------------
class TextSearchRequest(BaseModel):
    query: str
    language: str = "en"                         # "en" | "hi" | "gu"
    top_k: int = 60
    include_scenes: bool = True                  # include whole-frame 'scene' matches
    filters: SearchFilters = Field(default_factory=SearchFilters)


class PlateSearchRequest(BaseModel):
    plate: str                                   # full or partial
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ResultItem(BaseModel):
    detection_id: int
    camera_id: str
    camera_name: Optional[str] = None
    timestamp: Optional[str] = None
    class_label: str
    confidence: float
    score: float                                 # display relevance (0-1, relative within set)
    raw_score: Optional[float] = None            # raw CLIP cosine similarity
    crop_url: Optional[str] = None
    bbox: Optional[list[float]] = None           # [x, y, w, h] in native pixels
    attributes: dict = Field(default_factory=dict)
    track_id: Optional[int] = None
    # track-level de-duplication (one card per (camera, track); see result_utils)
    visible_from: Optional[str] = None           # first sighting timestamp of the track
    visible_until: Optional[str] = None          # last sighting timestamp of the track
    track_appearances: Optional[int] = None      # how many detections the track had
    # playback: jump straight to this moment in the source recording
    video_id: Optional[int] = None
    video_url: Optional[str] = None              # /media/videos/<clip>
    offset_seconds: Optional[float] = None        # seek position within the clip
    frame_width: Optional[int] = None            # native size (for bbox overlay)
    frame_height: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    translated_query: Optional[str] = None       # if translated from hi/gu
    total: int
    results: list[ResultItem]
    note: Optional[str] = None                   # e.g. "no strong match" guidance
    object_type: Optional[str] = None            # auto-inferred bias (person/vehicle)


# ------------------------- Cross-camera track ----------------------
class TrackAppearance(BaseModel):
    camera_id: str
    camera_name: Optional[str] = None
    timestamp: Optional[str] = None
    detection_id: int
    similarity: float
    crop_url: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    video_url: Optional[str] = None              # source clip for playback
    offset_seconds: Optional[float] = None        # seek position within the clip


class TrackSummary(BaseModel):
    total_appearances: int
    unique_cameras: int
    first_seen: Optional[str] = None             # ISO timestamp of earliest sighting
    last_seen: Optional[str] = None              # ISO timestamp of latest sighting
    span_seconds: Optional[float] = None         # last_seen - first_seen


class TrackResponse(BaseModel):
    reference_detection_id: int
    reference_class: Optional[str] = None        # "person" | vehicle label
    appearances: list[TrackAppearance]           # sorted by time
    summary: Optional[TrackSummary] = None


# ----------------------------- Ingestion ---------------------------
class IngestRequest(BaseModel):
    video: str                                   # filename within the server's video dir
    camera_id: Optional[str] = None
    start_time: Optional[str] = None             # ISO datetime of first frame
    fps: Optional[float] = None
    mode: Optional[str] = None                   # "fast" (default) | "accurate"; None -> config


class IngestResponse(BaseModel):
    video_id: int
    camera_id: str
    status: str
    message: Optional[str] = None


# ----------------------------- Forensics ---------------------------
class ExportRequest(BaseModel):
    detection_ids: list[int]
    case_number: str
    officer: str
    notes: Optional[str] = None


class ExportResponse(BaseModel):
    export_id: str
    manifest_hash: str
    download_url: str
    file_count: int


# ----------------------------- Audit -------------------------------
class AuditEntry(BaseModel):
    log_id: int
    timestamp: str
    action: str
    query_text: Optional[str] = None
    query_type: Optional[str] = None
    result_count: Optional[int] = None
    user: Optional[str] = None
