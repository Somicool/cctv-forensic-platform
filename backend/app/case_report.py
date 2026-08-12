"""Case File evidence report (PDF) for police investigation.

Builds a self-contained report for one case: cover sheet with case metadata, a
Gemini-written case overview, a chronological timeline, one full section per
exhibit (the original frame with the subject outlined, the close-up that was
matched, every recorded fact, and a situational description), a camera appendix
with GPS/siting details, and a SHA-256 chain-of-custody appendix.

This is ADDITIVE. The sealed ZIP export in forensics.py is untouched; this is the
readable document an officer files. Detection, tracking, search and re-ID are not
involved - the report only presents facts already stored.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import camera_registry, case_store, config, database, faces_gallery, gemini_report
from .search.text_search import _camera_names

_VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "auto-rickshaw",
                   "scooter", "tempo", "mini-truck", "pickup", "tractor", "hcv", "lcv"}


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_ts(ts) -> str:
    """'2026-08-06 18:57:39 (+05:30)' from a stored ISO timestamp."""
    if not ts:
        return "unknown"
    try:
        d = datetime.fromisoformat(str(ts))
    except ValueError:
        return str(ts)[:19]
    out = d.strftime("%d %b %Y, %H:%M:%S")
    if d.tzinfo is not None:
        off = d.strftime("%z")
        out += f" (UTC{off[:3]}:{off[3:]})" if off else ""
    return out


def _annotated_frame(det: dict, dest: Path) -> tuple[Path | None, str | None]:
    """The ORIGINAL camera frame with the subject outlined in red - the exact image
    the match was found in. Returns (path, how it was obtained).

    Prefers the sampled frame stored at ingest; falls back to decoding the frame
    from the source video. Box coordinates are scaled if the stored frame is not
    the same size as the source, so the outline always lands on the subject."""
    vid, fno = det.get("video_id"), det.get("frame_number")
    if det.get("bbox_x") is None:
        return None, None

    img, origin = None, None
    fpath = faces_gallery._frame_path(vid, fno)
    if fpath:
        img = cv2.imread(str(fpath))
        origin = "stored sampled frame"
    if img is None:
        vpath = faces_gallery.source_video_path(vid)
        if vpath:
            reader = faces_gallery._FrameReader(vpath)
            try:
                img = reader.read(fno)
                origin = "decoded from source recording"
            finally:
                reader.close()
    if img is None or not img.size:
        return None, None

    H, W = img.shape[:2]
    vmeta = database.video_index().get(vid) or {}
    src_w = vmeta.get("width") or W
    src_h = vmeta.get("height") or H
    sx = W / float(src_w or W)
    sy = H / float(src_h or H)

    x = int(det["bbox_x"] * sx); y = int(det["bbox_y"] * sy)
    w = int(det["bbox_w"] * sx); h = int(det["bbox_h"] * sy)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None, None

    out = img.copy()
    thick = max(2, int(round(min(W, H) / 400.0)))
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), thick)
    label = f"{det.get('class_label') or 'subject'} #{det.get('detection_id')}"
    fs = max(0.5, min(W, H) / 1100.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    ly = max(th + 6, y1)
    cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), (0, 0, 255), -1)
    cv2.putText(out, label, (x1 + 4, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), thick, cv2.LINE_AA)
    cv2.imwrite(str(dest), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return dest, origin


def _subject_image(det: dict, dest: Path) -> Path | None:
    """The close-up of the matched subject: the context-padded crop when one can be
    made, otherwise the stored search crop itself."""
    did = det["detection_id"]
    faces_gallery.expanded_crop_url(did)                    # builds + caches it
    for cand in (config.EXPANDED_CROP_DIR / f"exp_{did}.jpg", det.get("crop_path")):
        if cand and Path(cand).exists():
            shutil.copy2(cand, dest)
            return dest
    return None


def _fallback_description(det: dict, cam: dict | None) -> dict:
    """Description assembled from stored attributes, used when Gemini is off or
    unreachable. Deliberately states only recorded facts."""
    a = det.get("attributes") or {}
    label = det.get("class_label") or "subject"
    bits = []
    if a.get("color"):
        bits.append(f"{a['color']} in colour")
    if a.get("upper_color"):
        bits.append(f"upper clothing {a['upper_color']}")
    if a.get("lower_color"):
        bits.append(f"lower clothing {a['lower_color']}")
    if a.get("vehicle_type"):
        bits.append(f"identified as a {a['vehicle_type']}")
    if a.get("accessories"):
        acc = a["accessories"]
        bits.append("carrying/wearing " + (", ".join(acc) if isinstance(acc, list) else str(acc)))
    subject = f"Recorded as a {label}" + ((", " + "; ".join(bits)) if bits else "") + "."
    where = (cam or {}).get("location") or (cam or {}).get("address") or "an unspecified location"
    return {
        "scene": f"Captured by camera {det.get('camera_id')} at {where}.",
        "subject": subject,
        "actions": "Not assessed - automatic scene narration was unavailable for this exhibit.",
        "context": "Not assessed.",
        "quality": f"Detector confidence {round(float(det.get('confidence') or 0), 3)}.",
        "observations": [],
        "generated": "system (stored attributes only)",
    }


def _exhibit_facts(det: dict, cam: dict | None, vmeta: dict, cam_names: dict) -> dict:
    """Everything an investigator needs recorded for one exhibit."""
    a = det.get("attributes") or {}
    offset = None
    if det.get("frame_number") is not None and vmeta.get("native_fps"):
        offset = round(det["frame_number"] / vmeta["native_fps"], 2)
    return {
        "detection_id": det["detection_id"],
        "class": det.get("class_label"),
        "detector_confidence": round(float(det.get("confidence") or 0), 3),
        "track_id": det.get("track_id"),
        "camera_id": det.get("camera_id"),
        "camera_name": cam_names.get(det.get("camera_id")) or (cam or {}).get("name"),
        "camera_location": (cam or {}).get("location"),
        "camera_address": (cam or {}).get("address"),
        "camera_road": (cam or {}).get("road_name"),
        "camera_gps": (f"{(cam or {}).get('lat')}, {(cam or {}).get('lon')}"
                       if (cam or {}).get("lat") is not None else None),
        "camera_facing_deg": (cam or {}).get("facing_deg"),
        "recorded_at": det.get("timestamp"),
        "source_recording": vmeta.get("filename"),
        "recording_started": vmeta.get("start_time"),
        "frame_number": det.get("frame_number"),
        "offset_in_clip_seconds": offset,
        "bbox_xywh": ([round(float(det[k]), 1) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]
                      if det.get("bbox_x") is not None else None),
        "colour": a.get("color"),
        "upper_clothing": a.get("upper_color"),
        "lower_clothing": a.get("lower_color"),
        "vehicle_type": a.get("vehicle_type"),
        "accessories": a.get("accessories"),
        "plate_text": a.get("plate_text"),
        "plate_confidence": a.get("plate_confidence"),
    }


def build_case_report(case_key: str | None = None, case_info: dict | None = None,
                      use_gemini: bool = True, detection_ids: list[int] | None = None,
                      export_id: str | None = None) -> dict:
    """Assemble the PDF evidence report for one case. Returns a record dict.

    Two sources of exhibits:
      * `detection_ids` given  -> report on exactly those (used to produce the PDF
        for a case that was already sealed as an export, so an officer can pull the
        report for ANY past case, not only the one open in the workspace);
      * otherwise               -> the evidence currently saved against `case_key`.
    """
    ck = case_store._key(case_key)
    if detection_ids:
        evidence = [{"detection_id": int(i)} for i in detection_ids]
    else:
        evidence = case_store.list_evidence(ck)
    if not evidence:
        return {"error": "This case has no evidence yet. Add matches with the ＋ button first."}
    info = ({**(case_info or {})} if detection_ids
            else {**case_store.get_case_info(ck), **(case_info or {})})

    report_id = "RPT-" + uuid.uuid4().hex[:10]
    out_dir = config.EXPORT_DIR / report_id
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    cam_names = _camera_names()
    vindex = database.video_index()

    # Fresh facts from the database, ordered by real recording time. Snapshot is the
    # fallback so an exhibit never vanishes from a report if its row was re-ingested.
    ids = [e["detection_id"] for e in evidence]
    fresh = {d["detection_id"]: d for d in database.get_detections(ids)}
    exhibits = []
    for e in evidence:
        det = fresh.get(e["detection_id"]) or dict(e)
        det.setdefault("detection_id", e["detection_id"])
        exhibits.append(det)
    exhibits.sort(key=lambda d: str(d.get("timestamp") or ""))

    cameras_used: dict[str, dict] = {}
    prepared = []
    for i, det in enumerate(exhibits, 1):
        cid = det.get("camera_id")
        if cid and cid not in cameras_used:
            try:
                cameras_used[cid] = camera_registry.get_camera(cid) or {}
            except Exception:
                cameras_used[cid] = {}
        cam = cameras_used.get(cid) or {}
        vmeta = vindex.get(det.get("video_id")) or {}

        frame_path, frame_origin = _annotated_frame(det, img_dir / f"exhibit{i:02d}_frame.jpg")
        subj_path = _subject_image(det, img_dir / f"exhibit{i:02d}_subject.jpg")
        prepared.append({
            "index": i, "det": det, "cam": cam, "vmeta": vmeta,
            "frame": frame_path, "frame_origin": frame_origin, "subject": subj_path,
            "facts": _exhibit_facts(det, cam, vmeta, cam_names),
        })

    # --- Gemini narration (optional, fails soft) ---------------------------
    gem_used, gem_model, gem_ok = False, None, 0
    gem_error = None
    if use_gemini and gemini_report.available():
        gem_used = True
        results = gemini_report.describe_exhibits(
            [{"frame": p["frame"], "subject": p["subject"], "facts": p["facts"]}
             for p in prepared])
        for p, (d, err) in zip(prepared, results):
            if d:
                d.setdefault("generated", f"Gemini ({d.get('model')})")
                gem_model = d.get("model") or gem_model
                gem_ok += 1
                p["desc"] = d
            else:
                gem_error = gem_error or err
                p["desc"] = _fallback_description(p["det"], p["cam"])
    else:
        gem_error = "narration not requested" if not use_gemini else "narration unavailable"
        for p in prepared:
            p["desc"] = _fallback_description(p["det"], p["cam"])

    summary, summary_err = None, None
    if gem_used:
        summary, summary_err = gemini_report.summarise_case(
            {"case_title": info.get("title"), "case_number": info.get("caseNumber"),
             "officer": info.get("officer"), "notes": info.get("notes"),
             "exhibit_count": len(prepared)},
            [{"exhibit": p["index"], "facts": p["facts"],
              "scene": p["desc"].get("scene"), "subject": p["desc"].get("subject"),
              "actions": p["desc"].get("actions")} for p in prepared])
        gem_error = gem_error or summary_err

    created_at = datetime.now(timezone.utc).astimezone().isoformat()
    persons = sum(1 for p in prepared if (p["det"].get("class_label") or "") == "person")
    vehicles = sum(1 for p in prepared
                   if (p["det"].get("class_label") or "").lower() in _VEHICLE_LABELS)
    plates = sorted({(p["det"].get("attributes") or {}).get("plate_text")
                     for p in prepared if (p["det"].get("attributes") or {}).get("plate_text")})

    meta = {
        "report_id": report_id, "case_key": ck, "created_at": created_at,
        "case_info": info, "exhibit_count": len(prepared),
        "persons": persons, "vehicles": vehicles, "plates": plates,
        "cameras": cameras_used, "gemini_used": gem_used, "gemini_model": gem_model,
        "gemini_described": gem_ok, "gemini_error": gem_error, "summary": summary,
    }

    pdf_path = out_dir / "evidence_report.pdf"
    _render_pdf(pdf_path, meta, prepared)

    # Chain of custody over every file the report ships with.
    files = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(out_dir)).replace("\\", "/")] = _sha256_file(p)
    manifest = {**{k: v for k, v in meta.items() if k != "cameras"},
                "files": files}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report_sha = _sha256_file(pdf_path)

    rec = {
        "report_id": report_id,
        "case_key": ck,
        "export_id": export_id,
        "detection_ids": [p["det"]["detection_id"] for p in prepared],
        "case_number": info.get("caseNumber"),
        "case_title": info.get("title"),
        "officer": info.get("officer"),
        "created_at": created_at,
        "exhibit_count": len(prepared),
        "gemini_used": gem_used,
        "gemini_model": gem_model,
        "gemini_described": gem_ok,
        "gemini_error": gem_error,
        "sha256": report_sha,
        "file_path": str(pdf_path),
        "download_url": f"/media/exports/{report_id}/evidence_report.pdf",
    }
    _store_report(rec)
    database.log_audit("case_report", query_type="pdf", result_count=len(prepared),
                       details={"report_id": report_id, "case_number": info.get("caseNumber"),
                                "sha256": report_sha, "gemini": gem_used})
    return rec


# --------------------------------------------------------------- persistence
def _store_report(rec: dict) -> None:
    with database.get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO case_reports (report_id, case_key, export_id, detection_ids, "
            " case_number, case_title, officer, created_at, exhibit_count, gemini_used, "
            " gemini_model, sha256, file_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec["report_id"], rec["case_key"], rec.get("export_id"),
             json.dumps(rec.get("detection_ids") or []), rec["case_number"],
             rec["case_title"], rec["officer"], rec["created_at"], rec["exhibit_count"],
             1 if rec["gemini_used"] else 0, rec["gemini_model"], rec["sha256"],
             rec["file_path"]))


def build_report_for_export(export_id: str, use_gemini: bool = True) -> dict:
    """Evidence report for a case that was already sealed as an export.

    An export row IS a past case: it holds the case number, the officer and the exact
    detections that were filed. This lets an officer pull the PDF for any of those
    cases later, not just the one currently open in the workspace. Notes are
    recovered from the export's own manifest when it is still on disk."""
    row = next((e for e in database.list_exports() if e["export_id"] == export_id), None)
    if row is None:
        return {"error": f"Export {export_id} not found."}
    ids = row.get("detection_ids") or []
    if not ids:
        return {"error": f"Export {export_id} lists no detections to report on."}

    notes = None
    try:                                     # manifest.json sits beside the sealed zip
        man = config.EXPORT_DIR / export_id / "manifest.json"
        if man.exists():
            notes = json.loads(man.read_text(encoding="utf-8")).get("notes")
    except Exception:
        notes = None

    return build_case_report(
        case_key=f"export:{export_id}",
        case_info={"title": row.get("case_number") or export_id,
                   "caseNumber": row.get("case_number"),
                   "officer": row.get("officer"), "notes": notes},
        use_gemini=use_gemini, detection_ids=ids, export_id=export_id)


def list_reports(case_key: str | None = None) -> list[dict]:
    q = "SELECT * FROM case_reports"
    params: tuple = ()
    if case_key:
        q += " WHERE case_key=?"
        params = (case_store._key(case_key),)
    q += " ORDER BY created_at DESC"
    with database.get_conn() as conn:
        rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    for r in rows:
        r["gemini_used"] = bool(r.get("gemini_used"))
        r["download_url"] = f"/media/exports/{r['report_id']}/evidence_report.pdf"
        r["available"] = bool(r.get("file_path") and Path(r["file_path"]).exists())
        try:
            r["detection_ids"] = json.loads(r.get("detection_ids") or "[]")
        except (TypeError, ValueError):
            r["detection_ids"] = []
        # Convenience for the Evidence Gallery, where a report covers ONE exhibit.
        r["single_detection_id"] = (r["detection_ids"][0]
                                    if len(r["detection_ids"]) == 1 else None)
    return rows


def latest_report_by_export() -> dict[str, dict]:
    """{export_id: newest report for it} so the UI can offer a direct download
    instead of regenerating a report that already exists."""
    out: dict[str, dict] = {}
    for r in list_reports():
        eid = r.get("export_id")
        if eid and r.get("available") and eid not in out:
            out[eid] = r
    return out


# --------------------------------------------------------------- PDF rendering
_INK = "#12182b"
_ACCENT = "#b3261e"


def _styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY

    s = getSampleStyleSheet()
    s.add(ParagraphStyle("RTitle", parent=s["Title"], fontSize=22, leading=26,
                         textColor=colors.HexColor(_INK), spaceAfter=2))
    s.add(ParagraphStyle("RSub", parent=s["Normal"], fontSize=10.5, leading=14,
                         textColor=colors.HexColor("#4a5468")))
    s.add(ParagraphStyle("H", parent=s["Heading2"], fontSize=13, leading=16,
                         textColor=colors.HexColor(_INK), spaceBefore=10, spaceAfter=5))
    s.add(ParagraphStyle("H3", parent=s["Heading3"], fontSize=10.5, leading=13,
                         textColor=colors.HexColor(_ACCENT), spaceBefore=7, spaceAfter=3))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=9.5, leading=13.5,
                         alignment=TA_JUSTIFY))
    s.add(ParagraphStyle("Small", parent=s["Normal"], fontSize=8, leading=10.5,
                         textColor=colors.HexColor("#4a5468")))
    s.add(ParagraphStyle("Mono", parent=s["Normal"], fontSize=7.2, leading=9,
                         fontName="Courier"))
    s.add(ParagraphStyle("Cell", parent=s["Normal"], fontSize=8, leading=10.5))
    s.add(ParagraphStyle("CellB", parent=s["Normal"], fontSize=8, leading=10.5,
                         fontName="Helvetica-Bold"))
    return s


def _esc(v) -> str:
    from xml.sax.saxutils import escape
    if v is None or v == "":
        return "—"
    return escape(str(v))


def _fit_image(path, max_w, max_h):
    """Image flowable scaled to fit a box while keeping aspect ratio."""
    from reportlab.platypus import Image
    try:
        img = cv2.imread(str(path))
        if img is None or not img.size:
            return None
        h, w = img.shape[:2]
        scale = min(max_w / float(w), max_h / float(h))
        return Image(str(path), width=w * scale, height=h * scale)
    except Exception:
        return None


def _kv_table(pairs, styles, col_w):
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib import colors
    rows = [[Paragraph(_esc(k), styles["CellB"]), Paragraph(_esc(v), styles["Cell"])]
            for k, v in pairs]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9cfdd")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef1f7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _render_pdf(pdf_path, meta: dict, prepared: list[dict]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, PageBreak, KeepTogether)

    s = _styles()
    info = meta["case_info"]
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=16 * mm,
        title=f"Evidence Report {meta['report_id']}",
        author="NiriXan AI Forensic Investigation Platform",
        subject=f"Case {info.get('caseNumber') or '—'}")
    W = doc.width

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7488"))
        canvas.drawString(16 * mm, 10 * mm,
                          f"{meta['report_id']}  ·  Case {info.get('caseNumber') or '—'}"
                          f"  ·  CONFIDENTIAL - for authorised investigative use only")
        canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {d.page}")
        canvas.setStrokeColor(colors.HexColor("#c9cfdd"))
        canvas.line(16 * mm, 13 * mm, A4[0] - 16 * mm, 13 * mm)
        canvas.restoreState()

    st = []

    # ---------------- cover ----------------
    st.append(Paragraph("NiriXan AI · CCTV Evidence Report", s["RTitle"]))
    st.append(Paragraph("Automated forensic video investigation record", s["RSub"]))
    st.append(Spacer(1, 10))
    st.append(_kv_table([
        ("Case title", info.get("title")),
        ("Case number", info.get("caseNumber")),
        ("Lead officer", info.get("officer")),
        ("Report reference", meta["report_id"]),
        ("Report generated", _fmt_ts(meta["created_at"])),
        ("Exhibits", f"{meta['exhibit_count']} "
                     f"({meta['persons']} person, {meta['vehicles']} vehicle)"),
        ("Cameras involved", ", ".join(meta["cameras"].keys()) or "—"),
        ("Number plates recorded", ", ".join(meta["plates"]) or "None"),
    ], s, [42 * mm, W - 42 * mm]))

    if info.get("notes"):
        st.append(Spacer(1, 8))
        st.append(Paragraph("Investigating officer's notes", s["H3"]))
        st.append(Paragraph(_esc(info["notes"]), s["Body"]))

    # ---------------- case summary ----------------
    summ = meta.get("summary")
    st.append(Spacer(1, 6))
    st.append(Paragraph("1. Case summary", s["H"]))
    if summ:
        for label, key in (("Overview", "overview"), ("Movement across cameras", "movement"),
                           ("Corroboration", "corroboration")):
            if summ.get(key):
                st.append(Paragraph(label, s["H3"]))
                st.append(Paragraph(_esc(summ[key]), s["Body"]))
        for label, key in (("Recommended follow-up actions", "followups"),
                           ("Limitations of this evidence", "limitations")):
            vals = summ.get(key) or []
            if isinstance(vals, str):
                vals = [vals]
            if vals:
                st.append(Paragraph(label, s["H3"]))
                for v in vals:
                    st.append(Paragraph(f"•&nbsp;&nbsp;{_esc(v)}", s["Body"]))
    else:
        why = f" ({meta['gemini_error']})" if meta.get("gemini_error") else ""
        st.append(Paragraph(
            f"{meta['exhibit_count']} exhibit(s) collected from "
            f"{len(meta['cameras'])} camera(s). An automatic narrative summary was not "
            f"available for this report{why}; every exhibit below still carries its "
            "images and full recorded particulars.", s["Body"]))

    # ---------------- timeline ----------------
    st.append(Paragraph("2. Chronological timeline", s["H"]))
    rows = [[Paragraph(h, s["CellB"]) for h in
             ("#", "Recorded date &amp; time", "Camera", "Location", "Subject", "Plate")]]
    for p in prepared:
        d, cam = p["det"], p["cam"]
        a = d.get("attributes") or {}
        rows.append([
            Paragraph(str(p["index"]), s["Cell"]),
            Paragraph(_esc(_fmt_ts(d.get("timestamp"))), s["Cell"]),
            Paragraph(_esc(d.get("camera_id")), s["Cell"]),
            Paragraph(_esc(cam.get("location") or cam.get("address")), s["Cell"]),
            Paragraph(_esc(d.get("class_label")), s["Cell"]),
            Paragraph(_esc(a.get("plate_text")), s["Cell"]),
        ])
    tl = Table(rows, colWidths=[8 * mm, 38 * mm, 27 * mm, 40 * mm, 22 * mm,
                                W - 135 * mm], repeatRows=1)
    tl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_INK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9cfdd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f5fa")]),
    ]))
    st.append(tl)

    # ---------------- exhibits ----------------
    for p in prepared:
        st.append(PageBreak())
        d, cam, vmeta, desc = p["det"], p["cam"], p["vmeta"], p["desc"]
        st.append(Paragraph(f"Exhibit {p['index']} — {_esc(d.get('class_label'))} "
                            f"at {_esc(d.get('camera_id'))}", s["H"]))
        st.append(Paragraph(f"Recorded {_esc(_fmt_ts(d.get('timestamp')))} · "
                            f"detection ID {d.get('detection_id')} · "
                            f"track {_esc(d.get('track_id'))}", s["Small"]))
        st.append(Spacer(1, 6))

        # images: the searched frame (subject outlined) + the matched close-up
        frame_img = _fit_image(p["frame"], W * 0.62, 82 * mm) if p["frame"] else None
        subj_img = _fit_image(p["subject"], W * 0.32, 82 * mm) if p["subject"] else None
        cap_l = Paragraph("Original frame the subject was found in — subject outlined in red"
                          + (f" ({_esc(p['frame_origin'])})" if p.get("frame_origin") else ""),
                          s["Small"])
        cap_r = Paragraph("Matched image of the subject", s["Small"])
        if frame_img or subj_img:
            grid = Table([[frame_img or Paragraph("Frame unavailable", s["Small"]),
                           subj_img or Paragraph("Crop unavailable", s["Small"])],
                          [cap_l, cap_r]],
                         colWidths=[W * 0.65, W * 0.35])
            grid.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, 0), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 1), (-1, 1), 3),
            ]))
            st.append(grid)
            st.append(Spacer(1, 8))

        st.append(Paragraph("Situational description", s["H3"]))
        for label, key in (("Scene", "scene"), ("Subject", "subject"),
                           ("Apparent activity", "actions"),
                           ("Surroundings of note", "context"),
                           ("Image quality", "quality")):
            if desc.get(key):
                st.append(Paragraph(f"<b>{label}.</b> {_esc(desc[key])}", s["Body"]))
        obs = desc.get("observations") or []
        if isinstance(obs, str):
            obs = [obs]
        if obs:
            st.append(Paragraph("Points flagged for the investigator", s["H3"]))
            for o in obs:
                st.append(Paragraph(f"•&nbsp;&nbsp;{_esc(o)}", s["Body"]))
        st.append(Paragraph(f"Description source: {_esc(desc.get('generated'))}. "
                            "Machine-generated text requires verification by the "
                            "investigating officer before evidential use.", s["Small"]))

        st.append(Spacer(1, 7))
        st.append(Paragraph("Recorded particulars", s["H3"]))
        a = d.get("attributes") or {}
        pairs = [
            ("Camera", f"{d.get('camera_id')}"
                       + (f" ({cam.get('name')})" if cam.get("name") else "")),
            ("Location", cam.get("location") or cam.get("address")),
            ("Road / area", cam.get("road_name")),
            ("Camera GPS", (f"{cam.get('lat')}, {cam.get('lon')}"
                            if cam.get("lat") is not None else None)),
            ("Camera facing", (f"{cam.get('facing_deg')}° "
                               f"{camera_registry.compass_name(cam.get('facing_deg')) or ''}"
                               if cam.get("facing_deg") is not None else None)),
            ("Date &amp; time recorded", _fmt_ts(d.get("timestamp"))),
            ("Source recording", vmeta.get("filename")),
            ("Recording start", _fmt_ts(vmeta.get("start_time"))),
            ("Position in recording", (f"frame {d.get('frame_number')}"
                                       + (f" ≈ {p['facts']['offset_in_clip_seconds']}s in"
                                          if p["facts"].get("offset_in_clip_seconds") is not None
                                          else ""))),
            ("Subject class", d.get("class_label")),
            ("Detector confidence", p["facts"]["detector_confidence"]),
            ("Tracking ID", d.get("track_id")),
            ("Bounding box (x,y,w,h)", p["facts"].get("bbox_xywh")),
            ("Primary colour", a.get("color")),
            ("Upper clothing", a.get("upper_color")),
            ("Lower clothing", a.get("lower_color")),
            ("Vehicle type", a.get("vehicle_type")),
            ("Accessories", (", ".join(a["accessories"])
                             if isinstance(a.get("accessories"), list) else a.get("accessories"))),
            ("Number plate", a.get("plate_text")),
        ]
        st.append(_kv_table([(k, v) for k, v in pairs], s, [44 * mm, W - 44 * mm]))

    # ---------------- cameras appendix ----------------
    st.append(PageBreak())
    st.append(Paragraph("Appendix A — Cameras referenced", s["H"]))
    rows = [[Paragraph(h, s["CellB"]) for h in
             ("Camera ID", "Name", "Location / address", "GPS", "Facing", "Coverage")]]
    for cid, cam in meta["cameras"].items():
        rows.append([
            Paragraph(_esc(cid), s["Cell"]),
            Paragraph(_esc(cam.get("name")), s["Cell"]),
            Paragraph(_esc(cam.get("address") or cam.get("location")), s["Cell"]),
            Paragraph(_esc(f"{cam.get('lat')}, {cam.get('lon')}"
                           if cam.get("lat") is not None else None), s["Cell"]),
            Paragraph(_esc(f"{cam.get('facing_deg')}°"
                           if cam.get("facing_deg") is not None else None), s["Cell"]),
            Paragraph(_esc(f"{cam.get('coverage_m')} m"
                           if cam.get("coverage_m") is not None else None), s["Cell"]),
        ])
    ct = Table(rows, colWidths=[26 * mm, 26 * mm, W - 128 * mm, 32 * mm, 16 * mm, 20 * mm],
               repeatRows=1)
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_INK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9cfdd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    st.append(ct)
    st.append(Paragraph(
        "Cameras without recorded GPS coordinates limit map-based journey "
        "reconstruction. Coordinates can be completed in the Camera Registry.", s["Small"]))

    # ---------------- integrity + provenance ----------------
    st.append(Paragraph("Appendix B — Provenance and integrity", s["H"]))
    st.append(_kv_table([
        ("Report reference", meta["report_id"]),
        ("Generated", _fmt_ts(meta["created_at"])),
        ("Generated by", "NiriXan AI Forensic Investigation Platform (automated)"),
        ("Exhibits", meta["exhibit_count"]),
        ("Scene narration", (f"Gemini {meta.get('gemini_model')} — "
                             f"{meta.get('gemini_described')}/{meta['exhibit_count']} exhibits"
                             if meta.get("gemini_described")
                             else "Not available (stored attributes only)")),
        *([("Narration unavailable because",
            f"{meta['gemini_error']}. Recorded particulars below are unaffected.")]
          if meta.get("gemini_error") and not meta.get("gemini_described") else []),
        ("Accompanying files", "manifest.json lists a SHA-256 hash for every image "
                               "shipped with this report"),
    ], s, [44 * mm, W - 44 * mm]))
    st.append(Spacer(1, 8))
    st.append(Paragraph("Statement on automated content", s["H3"]))
    st.append(Paragraph(
        "Detections, timestamps, camera details and attributes in this report are "
        "recorded automatically by the video analysis pipeline. Where indicated, the "
        "situational descriptions were produced by a vision-language model from the "
        "frame shown alongside them. Those descriptions are investigative aids only: "
        "they do not identify any individual, do not establish that an offence took "
        "place, and must be verified against the source recording by the investigating "
        "officer before being relied upon. Timestamps reflect each recording's start "
        "time as derived from its file metadata.", s["Body"]))

    doc.build(st, onFirstPage=footer, onLaterPages=footer)
