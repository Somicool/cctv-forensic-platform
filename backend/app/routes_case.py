"""Case File endpoints - persistent evidence set + case metadata.

Additive: the search / journey / tracking pipelines are untouched. These routes
only give the Case File the same durability the rest of the platform already has.
"""
from __future__ import annotations

from fastapi import APIRouter, Body

from fastapi import HTTPException

from . import case_report, case_store

router = APIRouter()


@router.get("/case")
def get_case(case_key: str = case_store.DEFAULT_CASE):
    """Restore a saved investigation: evidence set + case details."""
    return case_store.load_case(case_key)


@router.get("/case/evidence")
def get_case_evidence(case_key: str = case_store.DEFAULT_CASE):
    return case_store.list_evidence(case_key)


@router.put("/case/evidence")
def put_case_evidence(payload: dict = Body(...)):
    """Persist the current evidence set (write-through from the frontend)."""
    return case_store.set_evidence(payload.get("items") or [], payload.get("case_key"))


@router.post("/case/evidence")
def post_case_evidence(payload: dict = Body(...)):
    """Append one evidence item."""
    return case_store.add_evidence(payload.get("item") or payload, payload.get("case_key"))


@router.delete("/case/evidence/{detection_id}")
def delete_case_evidence(detection_id: int, case_key: str = case_store.DEFAULT_CASE):
    return case_store.remove_evidence(detection_id, case_key)


@router.delete("/case/evidence")
def clear_case_evidence(case_key: str = case_store.DEFAULT_CASE):
    return case_store.clear_evidence(case_key)


@router.put("/case/info")
def put_case_info(payload: dict = Body(...)):
    """Persist case title / number / officer / notes."""
    info = payload.get("case_info") or payload.get("info") or payload
    return case_store.save_case_info(info, payload.get("case_key"))


@router.post("/case/report")
def create_case_report(payload: dict | None = None):
    """Build the PDF evidence report for a case.

    Includes, per exhibit, the original frame the subject was found in (subject
    outlined), the matched close-up, a Gemini-written situational description, and
    every recorded particular an investigator needs. Gemini narration can be turned
    off per request with {"use_gemini": false}; the report still builds from stored
    attributes."""
    payload = payload or {}
    rec = case_report.build_case_report(
        case_key=payload.get("case_key"),
        case_info=payload.get("case_info"),
        use_gemini=payload.get("use_gemini", True),
        detection_ids=payload.get("detection_ids"))
    if rec.get("error"):
        raise HTTPException(status_code=400, detail=rec["error"])
    return rec


@router.post("/case/report/for-export/{export_id}")
def create_report_for_export(export_id: str, payload: dict | None = None):
    """Build the PDF report for a case that was already sealed as an export.

    Lets an officer download the report for ANY past case, using that export's own
    case number, officer and filed detections."""
    payload = payload or {}
    rec = case_report.build_report_for_export(
        export_id, use_gemini=payload.get("use_gemini", True))
    if rec.get("error"):
        raise HTTPException(status_code=404, detail=rec["error"])
    return rec


@router.get("/case/reports")
def get_case_reports(case_key: str | None = None):
    """Previously generated evidence reports, newest first."""
    return case_report.list_reports(case_key)
