# AI-Driven Intelligent Descriptive Search for Smart City CCTV

**Hackathon:** ERH26_PS_07 · 4-week prototype

Describe a person, vehicle, or scene in plain language (English, Hindi, or Gujarati)
or upload an image, and get back matching, timestamped CCTV footage across cameras.
Built to **generalise to unfamiliar footage** the system has never seen.

> Example: *"a white truck"*, *"a red hatchback near Diamond Market"*,
> *"a person wearing a backpack"*, or Hindi *"सफ़ेद ट्रक"* → ranked, timestamped
> detections with crop thumbnails, filters, cross-camera tracking, and a
> tamper-evident forensic export.

---

## What it does

- **Descriptive text search** – natural-language queries over ingested video (CLIP semantic retrieval).
- **Multi-language** – Hindi / Gujarati queries auto-translated to English before search (offline fallback included).
- **Image & person re-ID search** – upload an image; find visually similar detections or the same person (OSNet).
- **Cross-camera tracking** – trace an entity's journey across cameras on a map + timeline.
- **Face search** (bonus, ethics-gated) – find a face across footage, with age/gender.
- **License-plate search** (bonus) – OCR plate text on vehicles, search by full/partial plate.
- **Metadata filters** – camera, time range, object type, colour, vehicle type, min confidence.
- **Forensic export** – zipped evidence package (crops + manifest + PDF) sealed with a **SHA-256 chain of custody**, plus a full audit log.
- **Live ingestion** – ingest new footage via the API with real-time WebSocket progress; unknown cameras are registered automatically.

## Architecture

Lean **6-model** stack (chosen for reliability on a 4 GB GPU):

| Stage | Model |
|---|---|
| Object detection | YOLOv10 |
| Tracking | ByteTrack |
| Semantic embeddings + zero-shot attributes | OpenCLIP ViT-B/16 |
| Person re-ID | OSNet (torchreid) |
| Face recognition (bonus) | InsightFace `buffalo_l` (CPU) |
| License-plate OCR (bonus) | EasyOCR (CPU) |

- **Backend:** FastAPI + FAISS (cosine/IP vector search) + SQLite (metadata) + static `/media`.
- **Frontend:** React + Vite dashboard (dark "command centre" theme, Leaflet map).
- **Vector/metadata join:** each FAISS id is the SQLite `detection_id` (or `face_id`), so search returns DB rows directly.

```
video → frames → YOLO detect + ByteTrack → crops
      → CLIP embed (crops + whole "scene" frames) + zero-shot attributes
      → OSNet embed (persons) → InsightFace (faces) → EasyOCR (plates)
      → SQLite rows + FAISS vectors
query → (translate) → CLIP text embed → FAISS search → filter → ranked results
```

## Repository layout

```
backend/
  app/
    main.py            FastAPI app: routes, /media static, /ws ingest progress
    routes.py          REST endpoints
    config.py          paths, model names, thresholds, vocabularies
    database.py        SQLite schema + helpers
    forensics.py       SHA-256 evidence export
    ingest_jobs.py     in-memory ingest job registry (WebSocket progress)
    models/schemas.py  Pydantic API contract
    ingestion/         video_processor, detector, tracker, embedder,
                       attribute_extractor, reid_embedder, face_recognizer,
                       plate_reader, pipeline
    search/            text_search, image_search, cross_camera, face_search,
                       plate_search, translate, filters, vector_store
  scripts/             ingest + verification tooling (see below)
  data/                cctv.db, faiss_indexes/, crops/, frames/, videos/, exports/
frontend/
  src/App.jsx, src/api.js, src/components/*  (Dashboard, Filters, ResultsGrid,
    ResultDetail, TrackingView, ForensicsView, CamerasView, AuditView)
PROGRESS.md            detailed build log / resume notes
```

## Setup

Prerequisites: Python 3.12, Node.js 18+ (tested on 24), an NVIDIA GPU is optional
(falls back to CPU).

```cmd
:: 1) Python env + PyTorch (CUDA 12.1 build; use the CPU index-url if no GPU)
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

:: 2) Cache the model weights (one-time; enables offline runs)
.venv\Scripts\python.exe backend\scripts\warmup_models.py

:: 3) Frontend deps
cd frontend && npm install
```

## Running

```cmd
:: Backend  (from backend/)
..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
::  -> API docs at http://localhost:8000/docs , health at /api/health

:: Frontend (from frontend/)  -> http://localhost:5173
npm run dev
```

The Vite dev server proxies `/api`, `/media`, and `/ws` to the backend on `:8000`.

### Ingesting footage

```cmd
:: Batch-ingest every clip in backend/data/videos
.venv\Scripts\python.exe backend\scripts\ingest_all.py --reset --start-time "2026-07-07T20:00:00"

:: Or live, via the API (progress streams over /ws/ingest/{job_id})
POST /api/ingest   { "video": "CAM-01_traffic.mp4", "camera_id": "CAM-01" }
```

## API (selected)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | device + camera count |
| GET  | `/api/cameras`, `/api/videos` | registry |
| POST | `/api/search/text` | descriptive search (`language`, `filters`, `include_scenes`) |
| POST | `/api/search/image` | image / person re-ID search (multipart) |
| POST | `/api/search/face` | face search (multipart) |
| POST | `/api/search/plate` | plate search |
| GET  | `/api/track/{detection_id}` | cross-camera appearances |
| POST | `/api/ingest` + WS `/ws/ingest/{job_id}` | live ingest + progress |
| POST | `/api/export`, GET `/api/exports` | forensic export |
| GET  | `/api/audit` | audit trail |

## Demo queries (verified to work well on the sample data)

- Text: **"a white truck"**, **"a car"** (filter camera = CAM-01 → surfaces a red hatchback), **"a person wearing a backpack"**, **"a crowded plaza"**
- Hindi: **"सफ़ेद ट्रक"** → translates to *white truck* → same footage as the English query
- Image: upload any crop → visually similar detections (toggle *Person re-ID* for people)
- Tracking: open a person result → **Track across cameras** / **Open in map view**
- Forensics: add results to the case file → export → download the SHA-256-sealed `.zip`

## Verification

Every capability has an automated check. Run them all in one command:

```cmd
:: from backend/  (set UTF-8 so Hindi/Gujarais print on Windows)
set PYTHONUTF8=1
..\.venv\Scripts\python.exe -u scripts\run_all_checks.py
```

Latest full sweep: **54 pass / 0 fail across 6 sections — ALL GREEN**
(baseline ingest + integrity, plate OCR, translation, forensic export, robustness/
unfamiliar-footage, REST API + WebSocket). Individual suites also runnable:
`reingest_and_verify.py`, `verify_plates.py`, `verify_translate.py`,
`verify_forensics.py`, `verify_robustness.py`, `verify_api.py`.
Frontend: `npm run build` (compile check).

## Known limitations (honest)

- **Attribute accuracy:** CLIP zero-shot colour/type is imperfect on small/distant crops; best on large, clear crops.
- **Re-ID weights:** OSNet uses generic ImageNet weights (permissive at threshold 0.75). Market-1501 weights would sharpen person re-ID (drop-in swap).
- **Cross-camera on sample data:** the bundled clips are separate scenes with no camera overlap, so a traced "journey" is mostly single-camera. The view, path line, and plumbing fully support real multi-camera footage.
- **License plates:** the sample clips contain no clearly readable plates (plate count = 0); the OCR + search path is proven on synthetic plates in `verify_plates.py`. Real plate-bearing footage yields real hits.
- **Scale:** FAISS uses an exact `IndexFlatIP` (linear scan) — ideal for a prototype's volumes. City-scale would swap in an ANN index (IVF/HNSW), a standard change.
- **Video playback:** `.avi` clips (CAM-02) don't decode in browsers; `.mp4` clips do. Frames/crops always display.
- **Translation:** online path uses Google via `deep-translator`; a small offline phrase dictionary covers common descriptive terms when there is no internet.

## Responsible use

Face recognition is a bonus capability gated behind `config.FACE_RECOGNITION_ENABLED`
and intended for authorised investigative use only. Every search and export is
written to an audit log, and exports carry a SHA-256 chain-of-custody seal.
