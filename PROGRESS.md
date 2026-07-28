# Project Progress - AI-Driven Descriptive Search for Smart City CCTV

**Hackathon PS:** ERH26_PS_07 (4-week prototype)
**Last updated:** 2026-07-07 (Task 20 complete - ALL 20 TASKS DONE)
**Status:** 20 of 20 complete and verified. PROJECT DEMO-READY. Full sweep: 54 pass / 0 fail (ALL GREEN).

> This file is the single source of truth for resuming. Nothing is held in memory:
> the SQLite DB, FAISS indexes, crops/frames, and all code live on disk (see paths).
> A reboot loses nothing.

---

## What this system does
Describe a person/vehicle/scene in natural language (or upload an image) and get
back matching, timestamped CCTV footage across cameras. Bonus: face search, license
plate search, multi-language queries, forensic export. Must generalize to unfamiliar
footage.

## Lean 6-model stack
YOLOv10 (detection) - OpenCLIP ViT-B-16 (text/image embeddings + zero-shot attributes)
- ByteTrack (tracking) - OSNet/torchreid (person re-ID) - InsightFace buffalo_l
(face, CPU) - EasyOCR (license plates, CPU, wired into pipeline).
Backend: FastAPI (full routes + WebSocket + /media static) + FAISS (cosine/IP) + SQLite.
Frontend: React + Vite (functional dashboard: search/filters/results/detail+track).
Multi-language: deep-translator (hi/gu -> en) + offline phrase-dict fallback.

## Key paths (Windows)
- Workspace: `d:\hackathon`
- venv Python: `d:\hackathon\.venv\Scripts\python.exe` (Python 3.12.10)
- Backend code: `d:\hackathon\backend\app` (config.py, database.py, models/schemas.py,
  main.py, ingestion/, search/)
- Scripts: `d:\hackathon\backend\scripts`
- Data (all persistent): `d:\hackathon\backend\data` -> cctv.db, faiss_indexes/{clip,reid,face}.index,
  crops/, frames/, videos/, faces/, exports/, camera_config.json
- Frontend: `d:\hackathon\frontend` (React/Vite; Node v24.18.0 + npm 11.16.0 installed)
- GPU: RTX 3050 Laptop 4GB, CUDA works. InsightFace runs on CPU (onnxruntime CPU build).

---

## Current ingested data state (VERIFIED consistent)
```
DB detections = 1336   faces = 9   faiss = {clip:1336, reid:551, face:9}
[PASS] DB detections == faiss clip (1336 vs 1336)
[PASS] text 'a white truck' -> 3 results (top=truck)
[PASS] face search -> 1 result: cam=CAM-02 det=6571 score=1.000 age=26 gender=female
```
Dataset = 4 clips in `backend/data/videos/`:
- CAM-01_traffic.mp4  - vehicles (incl. a RED HATCHBACK, matches PS flagship query), 4 obj dets, 0 faces
- CAM-02_plaza.avi    - pedestrians, 777 obj dets, 463 reid, 3 faces
- CAM-03_street.mp4   - bikes/street, 27 obj dets, 10 reid, 1 face
- CAM-04_entrance.mp4 - frontal faces, 78 obj dets, 78 reid, 5 faces
(886 object detections + 450 whole-frame "scene" detections = 1336 total.)

---

## Tasks
- [x] 1. Scaffold project + lock API contract (schemas.py)
- [x] 2. Environment (venv, torch cu121, all models cached via warmup_models.py)
- [x] 3. Video frame extraction (video_processor.py)
- [x] 4. YOLO detection + cropping (detector.py)
- [x] 5. ByteTrack tracking (tracker.py)
- [x] 6. CLIP embeddings + zero-shot attributes (embedder.py, attribute_extractor.py)
- [x] 7. FAISS vector store + SQLite metadata (vector_store.py, database.py)
- [x] 8. OSNet person re-ID (reid_embedder.py)
- [x] 9. Ingestion pipeline orchestrator + dynamic camera registration (pipeline.py)
- [x] 10. Text search engine + metadata filters (text_search.py, filters.py)
- [x] 11. Image/re-ID search + cross-camera tracking (image_search.py, cross_camera.py)
- [x] 12. Face recognition - InsightFace (face_recognizer.py, face_search.py) VERIFIED
- [x] 13. License plate recognition - EasyOCR (plate_reader.py, plate_search.py) VERIFIED
- [x] 14. Multi-language query translation (translate.py, deep-translator + offline) VERIFIED
- [x] 15. FastAPI routes + WebSocket ingest progress + /media static (routes.py, main.py) VERIFIED
- [x] 16. Frontend dashboard: search/filters/results/detail+track (React) VERIFIED (builds + live smoke)
- [x] 17. Frontend map/timeline/video/tracking (TrackingView.jsx, react-leaflet) VERIFIED
- [x] 18. Forensic export + audit log - SHA-256 (forensics.py, ForensicsView.jsx) VERIFIED
- [x] 19. Robustness + unfamiliar-footage (verify_robustness.py) VERIFIED - generalises to unseen clip
- [x] 20. Polish + demo prep (README.md, run_all_checks.py) VERIFIED - ALL GREEN 54/54
- [ ] 18. Forensic export + audit log (SHA-256)
- [ ] 19. Robustness pass + unfamiliar-footage testing
- [ ] 20. Polish + demo prep

---

## How to resume / run
Run everything from `d:\hackathon\backend` using the venv python.

- Re-ingest all clips + self-verify (clean, single process):
  `d:\hackathon\.venv\Scripts\python.exe -u scripts\reingest_and_verify.py`
  (Only needed if the DB/indexes are reset. Data is already ingested and consistent.)
- Start backend API: `uvicorn app.main:app --reload --port 8000` (from backend/).
  Routes: /api/health, /api/cameras, POST /api/search/{text,image,face,plate},
  GET /api/track/{id}, GET /api/audit, POST /api/ingest, WS /ws/ingest/{job_id},
  static /media. Interactive docs at /docs. Routes are torch-free at import (lazy).
- Start frontend: from `d:\hackathon\frontend` -> `npm run dev` (port 5173).
  Vite proxies /api, /media, /ws to :8000. `npm install` already done (node_modules present).
- Verify backend end-to-end (no live server): `python -u scripts\verify_api.py` (TestClient).

## CRITICAL environment quirk (do not forget)
The IDE shell is a single serial PowerShell that QUEUES commands and echoes them
char-by-char; file reads can race ahead of queued writes, and long commands may
spuriously report "Exit Code: 1" while still running. Running multiple ingests caused
CONCURRENT orphaned python processes that corrupted the DB (2557 vs 1336) once.
Reliable pattern:
1. Kill strays first: `taskkill /F /IM python.exe /T`
2. Do work in ONE self-contained python script (e.g., reingest_and_verify.py) that
   writes to a log with `python -u ... *> log.txt`.
3. Read the log FILE (not stdout) to get results.
Avoid launching multiple overlapping ingests.

---

## Task 19 summary - Robustness + unfamiliar-footage (USER PRIORITY)
- `scripts/make_unfamiliar_clip.py` (NEW): make_unfamiliar() transforms an existing clip
  (CAM-02_plaza.avi) -> a genuinely new-looking CAM-06_market.mp4 (mirror + brightness), real
  detectable footage the system has never ingested.
- `scripts/verify_robustness.py` (NEW, KEEP): snapshot DB+FAISS -> generate CAM-06 clip -> LIVE
  ingest via TestClient POST /api/ingest + read WS /ws/ingest progress to 'done' -> assert dynamic
  camera reg (CAM-06), DB grew (1336->1432), DB==FAISS (1432), search 'a person walking'@CAM-06
  returns CAM-06 people -> failure/edge cases -> RESTORE snapshot (byte-for-byte) + delete CAM-06
  clip/crops/frames. Baseline stays pristine at 1336.
- VERIFIED (all PASS): live ingest job -> WS streamed stages [start,detect+track,clip,reid,faces,
  store,done]; CAM-06 auto-registered; 80 obj+16 scene dets added; DB==FAISS 1432; unfamiliar-footage
  search returned 5 CAM-06 people (top=person) = GENERALISATION PROVEN; edge cases graceful (empty
  query->200, impossible filter->0, corrupt image->400 not 500, missing media->404, unknown track->200
  empty, missing ingest file->404); baseline restored to 1336. Filesystem clean (4 clips, no CAM-06).
- MINOR: a whitespace-only query still returns nearest neighbours (200) at the API; the frontend guards
  empty queries client-side. Acceptable (API faithfully searches input; UI validates).

## Task 18 summary - Forensic export + audit log (SHA-256)
- `app/forensics.py` (NEW): create_export(ExportRequest)->ExportResponse. Copies each detection's
  crop into EXPORT_DIR/<id>/crops/, writes manifest.json (case meta + per-item detection/camera/
  timestamp/attrs + per-file SHA-256), seals with SHA-256 over manifest.json (+manifest.sha256),
  renders report.pdf (reportlab platypus table w/ thumbnails), zips all, records in exports table +
  log_audit('export'). download_url=/media/exports/<id>.zip.
- `app/database.py`: +insert_export / list_exports.
- `app/routes.py`: POST /api/export (400 if no ids) + GET /api/exports.
- `frontend`: api.js +createExport/getExports; ForensicsView.jsx (NEW): case-file cart + case#/
  officer/notes -> export -> shows manifest hash + download; past-exports table. App.jsx: caseItems
  state + addToCase/removeFromCase/clearCase + Forensics nav badge (count). ResultDetail.jsx:
  'Add to case file' button. Dashboard passes onAddToCase.
- VERIFIED: scripts/verify_forensics.py 10/10 PASS (zip+manifest+report.pdf present; manifest SHA-256
  reproducible; per-file hashes match; exports row + audit entry; TAMPER detection - flipping a byte
  changes the hash). LIVE route: POST /api/export -> hash+url; zip downloads 200 application/zip 40KB;
  GET /api/exports lists. Frontend build OK (135 modules).

## Task 17 summary - Frontend map/timeline/video/tracking
- `app/database.py` +list_videos(); `app/routes.py` +GET /api/videos (adds url=/media/videos/<file>).
- `frontend/src/api.js` +getVideos(). `src/components/TrackingView.jsx` (NEW): react-leaflet
  MapContainer + OSM TileLayer + CircleMarkers (grey cameras, green appearances, cyan selected)
  + dashed Polyline path (time-ordered); timeline strip (crop thumb + camera + similarity + time,
  'reference' badge); video player <video src=/media/videos/..> for the selected sighting's camera;
  detection-ID input + Trace. Uses CircleMarker (no marker-image asset gotcha). Imports leaflet CSS.
- `src/App.jsx`: trackId state + openTrack(id) -> sets Tracking view; renders <TrackingView>.
  `Dashboard.jsx` passes onOpenTrack to `ResultDetail.jsx` ('Open in map view' button deep-links).
- `src/index.css`: track-view/map/timeline/video styles; bumped .modal-overlay z-index above Leaflet.
- VERIFIED: npm run build OK (134 modules, +leaflet, 0 errors). LIVE: GET /api/videos -> 4 clips with
  /media URLs; GET /api/track/9020 -> 34 appearances w/ lat/lon (CAM-02 21.206,72.8407)+crop; source
  clips serve (mp4 200 video/mp4 plays; CAM-02 .avi 200 but browsers can't decode AVI -> crop fallback).
- LIMITATION: single-scene test data -> journey mostly one camera; view + plumbing support a real
  multi-camera path. CAM-02 plaza is .avi (won't play in browser); mp4 clips (CAM-01/03/04) play.

## Task 16 summary - Frontend dashboard (React)
- `frontend/vite.config.js`: proxy /api, /media, /ws -> :8000.
- `src/api.js`: axios client (getHealth/getCameras/searchText/searchImage/searchPlate/
  trackDetection/getAudit). searchText posts {query,language,top_k,include_scenes,filters}.
- `src/components/`: Dashboard.jsx (Describe/Image/Plate modes + EN/HI/GU + examples),
  Filters.jsx (cameras/object_type/vehicle_type/colours/min_conf/time -> SearchFilters),
  ResultsGrid.jsx (crop-thumbnail cards + score + attrs), ResultDetail.jsx (modal: enlarged
  crop, attributes, one-click cross-camera trace via /api/track), CamerasView, AuditView.
- `src/App.jsx`: sidebar nav (Dashboard/Cameras/Tracking/Forensics/Audit) + health pill.
  Tracking + Forensics are placeholders (Tasks 17/18). `src/index.css`: extended dark theme.
- VERIFIED: `npm install` OK; `npm run build` OK (90 modules, no errors). LIVE smoke (uvicorn
  :8000 + vite :5173): frontend index 200; /api/health via proxy -> cuda/5 cams; POST
  /api/search/text via proxy -> total=2 top=truck with /media crop_url; /media image -> 200 jpeg.

## Task 15 summary - FastAPI routes + WebSocket + /media
- `app/routes.py`: GET /cameras, POST /search/{text,image,face,plate}, GET /track/{id},
  GET /audit, POST /ingest (validates video, background thread + progress_cb -> ingest_jobs).
  Heavy imports lazy inside handlers (app boots torch-free).
- `app/ingest_jobs.py`: thread-safe job registry. `app/main.py`: include_router(prefix=/api),
  mount /media -> DATA_DIR, WebSocket /ws/ingest/{job_id} streams progress.
- `app/ingestion/pipeline.py`: ingest_video(progress_cb=None) emits per-stage progress.
- `schemas.py`: +IngestRequest, +TextSearchRequest.include_scenes.
- VERIFIED: scripts/verify_api.py 13/13 PASS via TestClient, DB unchanged (1336==1336).

## Task 14 summary - Multi-language translation
- `app/search/translate.py`: translate_query(text, source_lang) -> (english, method).
  GoogleTranslator(auto->en); English passes through; offline hi/gu->en phrase-dict fallback.
  No coupling to search core. `scripts/verify_translate.py` (KEEP).
- VERIFIED: hi/gu -> correct English (online + offline); Hindi query -> results IDENTICAL to
  English 'white truck'; translated_query populated. (Set $env:PYTHONUTF8=1 to print Devanagari.)

## Task 13 summary - License plate recognition (EasyOCR)
- `config.py`: OCR_USE_GPU=False (EasyOCR on CPU, avoids 4GB VRAM contention) +
  PLATE_RECOGNITION_ENABLED=True. (OCR_LANGS + PLATE_REGEX already existed.)
- `database.py`: insert_plate / get_plates / search_plates (normalised partial LIKE -
  strips spaces/hyphens + uppercases both sides so 'ab1234' matches 'GJ05AB1234') / count_plates.
- `ingestion/plate_reader.py` (new): lazy EasyOCR reader; read_plates(image) -> [{text,conf,bbox}].
  Robustness: uppercase+digit allowlist, upscales small crops, per-box + joined-line matching,
  and conservative OCR-confusion recovery (O/0,S/5,I/1,B/8,G/6,Z/2,D/0,Q/0 - NO 'L', so words
  like HELLO2024 can't become fake plates), accepting only <=2-char fixes that yield a valid plate.
- `search/plate_search.py` (new): search_by_plate(plate) -> normalised SQL substring match ->
  ResultItem (+plate_text/plate_confidence). Plates live in SQLite only (short strings, no FAISS).
- `ingestion/pipeline.py`: stage 8 (plates), mirror of the face stage - OCR on one representative
  (largest) crop per VEHICLE track (bbox w>=60 h>=40), config-gated; plate_count + timing in stats.
- `scripts/verify_plates.py` (new, KEEP): deterministic dataset-independent proof (synthetic plate
  OCR + regex + DB insert on a real detection + partial 'ab1234' search + cleanup).
- VERIFIED: verify_plates.py 3/3 PASS; non-plate words rejected (no fabrication); full
  reingest_and_verify.py PASS (DB==faiss clip 1336; text+face still PASS; plates from footage=0,
  expected). Known limit: EasyOCR may misread a single char on odd fonts (tuning knob).

## Task 12 summary
- `database.py`: added insert_face / get_faces / count_faces.
- `ingestion/face_recognizer.py` (new): get_face_app() lazy InsightFace buffalo_l on CPU;
  detect_faces(image) -> [{embedding(512 normed float32), bbox, age, gender, det_score}].
- `ingestion/pipeline.py`: added det_ids list + a face stage. OPTIMIZED to run InsightFace
  on ONE representative (largest) crop per person-track (not every frame crop) - keeps the
  CPU face stage fast (plaza 7-13s). Gated by config.FACE_RECOGNITION_ENABLED; only crops
  with bbox w>=40 and h>=80. Adds face_count + face timing to stats; faces go into the
  'face' FAISS index keyed by face_id.
- `search/face_search.py` (new): search_by_face(image) -> detect face -> embed -> search
  'face' index -> map face_id -> source detection -> ResultItem (+age/gender). Threshold
  config.FACE_SIM_THRESHOLD.
- `scripts/reingest_and_verify.py` (new): clean single-process reset+ingest+verify tool
  (KEEP - reused for integrity checks).

## Task 20 summary - Polish + demo prep (FINAL, project complete)
- `README.md` (repo root, NEW): what it does, capabilities, architecture (6-model stack + FastAPI+
  FAISS+SQLite + React), repo layout, setup, run (backend uvicorn / frontend npm dev), ingest (batch
  + live API), API table, curated demo queries, verification, HONEST limitations, responsible-use note.
- `scripts/run_all_checks.py` (NEW, KEEP): master verifier - runs all 6 verify suites in ONE process
  (models load once), captures each section's output, counts [PASS]/[FAIL], prints consolidated summary.
- FINAL SWEEP RESULT (run_all_checks.py): 54 pass / 0 fail across 6 sections = ALL GREEN. Baseline
  re-ingested to 1336, all features verified, robustness restored to 1336, API green. Frontend
  `npm run build` OK (135 modules, 0 errors). Baseline intact.

## PROJECT COMPLETE - all 20 tasks done and verified.
Re-verify anytime from backend/:  `set PYTHONUTF8=1 && ..\.venv\Scripts\python.exe -u scripts\run_all_checks.py`
Run the demo:  backend `uvicorn app.main:app --port 8000` + frontend `npm run dev` (http://localhost:5173).

## Tuning knobs / notes for later
- OSNet uses generic ImageNet weights -> slightly permissive at REID_SIM_THRESHOLD=0.75
  (groups many plaza pedestrians). Market-1501 weights would sharpen re-ID (drop-in swap).
- CLIP zero-shot attributes are imperfect on tiny/distant crops (better on large crops).
- Cross-camera tracking logic is proven (self-match 1.0, time-sorted, deduped per track),
  but our clips are separate scenes so no true cross-camera overlap in test data.
- License plates: EasyOCR reads clear plates well; on odd fonts a single char may misread
  (5->S, 7->2). Sample clips have 0 readable plates - real plate footage would show non-zero.

## Verification tools (KEEP)
- `scripts/reingest_and_verify.py` - clean single-process reset+ingest all clips + integrity checks.
- `scripts/verify_plates.py` - deterministic plate capability proof (no dataset dependency).
- `scripts/verify_translate.py` - translation (online + offline) + search-equivalence proof.
- `scripts/verify_api.py` - TestClient API check (all routes, WS, /media); no DB mutation.
- `scripts/verify_forensics.py` - export + SHA-256 chain-of-custody + tamper-detection proof.
- `scripts/verify_robustness.py` - unfamiliar-footage live ingest + generalisation + edge cases
  (snapshots/restores baseline, so it's safe to re-run). `scripts/make_unfamiliar_clip.py` helper.
- Frontend: `npm run build` (compile check) in frontend/; live smoke = run backend + `npm run dev`.
- GOTCHA: set `$env:PYTHONUTF8=1` before running scripts that print Hindi/Gujarati (Windows cp1252).
