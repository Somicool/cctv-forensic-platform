"""Forensic evidence export with a SHA-256 chain of custody.

create_export(req) gathers the selected detections, copies their crop images into
an export folder, writes a manifest.json (case metadata + per-item facts + a
SHA-256 of every copied file), seals it with a SHA-256 over the manifest itself,
renders a PDF summary, and zips the lot. The export is recorded in the `exports`
table and written to the audit log. Any later tampering with an exported file
changes its hash and breaks the manifest seal.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import config, database
from .models.schemas import ExportRequest, ExportResponse
from .search.text_search import _camera_names


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def create_export(req: ExportRequest) -> ExportResponse:
    export_id = "EXP-" + uuid.uuid4().hex[:10]
    out_dir = config.EXPORT_DIR / export_id
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    dets = database.get_detections(req.detection_ids)
    cam_names = _camera_names()

    items = []
    for d in dets:
        crop_src = d.get("crop_path")
        crop_rel, crop_sha = None, None
        if crop_src and Path(crop_src).exists():
            fname = f"{d['detection_id']}_{Path(crop_src).name}"
            dst = crops_dir / fname
            shutil.copy2(crop_src, dst)
            crop_rel = f"crops/{fname}"
            crop_sha = _sha256_file(dst)
        items.append({
            "detection_id": d["detection_id"],
            "camera_id": d.get("camera_id"),
            "camera_name": cam_names.get(d.get("camera_id")),
            "timestamp": d.get("timestamp"),
            "class_label": d.get("class_label"),
            "confidence": d.get("confidence"),
            "track_id": d.get("track_id"),
            "attributes": d.get("attributes"),
            "crop_file": crop_rel,
            "crop_sha256": crop_sha,
        })

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "export_id": export_id,
        "case_number": req.case_number,
        "officer": req.officer,
        "notes": req.notes,
        "created_at": created_at,
        "generated_by": "NiriXan AI Forensic Investigation Platform",
        "item_count": len(items),
        "requested_detection_ids": list(req.detection_ids),
        "items": items,
    }
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    (out_dir / "manifest.json").write_bytes(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    # Chain-of-custody seal alongside the manifest.
    (out_dir / "manifest.sha256").write_text(f"{manifest_hash}  manifest.json\n", encoding="utf-8")

    try:
        _write_pdf(out_dir / "report.pdf", manifest, out_dir)
    except Exception as exc:  # noqa: BLE001 - PDF is a nice-to-have, never fail the export
        print(f"[forensics] PDF generation skipped: {exc}")

    zip_path = config.EXPORT_DIR / f"{export_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out_dir))

    file_count = sum(1 for p in out_dir.rglob("*") if p.is_file())

    database.insert_export(export_id, req.case_number, req.officer, created_at,
                           manifest_hash, str(zip_path), req.detection_ids)
    database.log_audit("export", query_type="forensic", result_count=len(items),
                       details={"export_id": export_id, "case_number": req.case_number,
                                "manifest_hash": manifest_hash})

    return ExportResponse(export_id=export_id, manifest_hash=manifest_hash,
                          download_url=f"/media/exports/{export_id}.zip",
                          file_count=file_count)


def _write_pdf(pdf_path, manifest, out_dir) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            title=f"Forensic Export {manifest['export_id']}")
    story = [Paragraph("NiriXan AI · Forensic Evidence Export", styles["Title"]),
             Spacer(1, 6)]
    for line in (f"Export ID: {manifest['export_id']}",
                 f"Case number: {manifest['case_number']}",
                 f"Officer: {manifest['officer']}",
                 f"Created (UTC): {manifest['created_at']}",
                 f"Items: {manifest['item_count']}"):
        story.append(Paragraph(line, styles["Normal"]))
    if manifest.get("notes"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Notes: {manifest['notes']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    rows = [["#", "Det ID", "Camera", "Time (UTC)", "Class", "Attributes", "Thumb"]]
    for i, it in enumerate(manifest["items"], 1):
        thumb = ""
        if it.get("crop_file"):
            p = out_dir / it["crop_file"]
            if p.exists():
                try:
                    thumb = Image(str(p), width=20 * mm, height=20 * mm)
                except Exception:
                    thumb = ""
        attrs = it.get("attributes") or {}
        attr_str = ", ".join(f"{k}:{v}" for k, v in attrs.items() if k != "kind")[:60]
        rows.append([str(i), str(it["detection_id"]), it.get("camera_id") or "",
                     (it.get("timestamp") or "")[:19], it.get("class_label") or "",
                     attr_str, thumb])
    table = Table(rows, colWidths=[8 * mm, 16 * mm, 20 * mm, 32 * mm, 18 * mm, 44 * mm, 22 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2233")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f7")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "System-generated evidence summary. Integrity is sealed by the SHA-256 manifest "
        "hash; modifying any exported file invalidates the hash.", styles["Italic"]))
    doc.build(story)
