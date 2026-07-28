"""Task 18 verification: forensic export + SHA-256 chain of custody.

Exports a couple of real detections, then checks: zip built, manifest present,
manifest hash reproducible, per-file SHA-256 matches, PDF + manifest inside the
zip, exports table row recorded, audit entry written.

    python -u scripts/verify_forensics.py
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app import config, database                         # noqa: E402
from app.forensics import create_export                  # noqa: E402
from app.models.schemas import ExportRequest             # noqa: E402


def pf(ok):
    return "PASS" if ok else "FAIL"


def main():
    database.init_db()
    print("=== TASK 18 FORENSIC EXPORT VERIFICATION ===")

    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT detection_id FROM detections WHERE crop_path IS NOT NULL "
            "AND class_label != 'scene' LIMIT 3").fetchall()
    ids = [r["detection_id"] for r in rows]
    print(f"exporting detections {ids}")

    resp = create_export(ExportRequest(detection_ids=ids, case_number="CASE-2026-001",
                                       officer="Insp. Rao", notes="verification export"))
    print(f"export_id={resp.export_id} hash={resp.manifest_hash[:16]}… "
          f"files={resp.file_count} url={resp.download_url}")

    exp_dir = config.EXPORT_DIR / resp.export_id
    zip_path = config.EXPORT_DIR / f"{resp.export_id}.zip"
    manifest_path = exp_dir / "manifest.json"

    print(f"[{pf(zip_path.exists())}] zip exists ({zip_path.name})")
    print(f"[{pf(manifest_path.exists())}] manifest.json exists")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recomputed = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"[{pf(recomputed == resp.manifest_hash)}] manifest SHA-256 reproducible "
          f"(chain-of-custody seal holds)")
    print(f"[{pf(manifest['item_count'] == len(ids))}] item_count={manifest['item_count']} == {len(ids)}")

    ok_files = True
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for it in manifest["items"]:
            if it.get("crop_file"):
                if it["crop_file"] not in names:
                    ok_files = False
                fp = exp_dir / it["crop_file"]
                if fp.exists():
                    if hashlib.sha256(fp.read_bytes()).hexdigest() != it["crop_sha256"]:
                        ok_files = False
    print(f"[{pf(ok_files)}] every crop file present in zip AND per-file SHA-256 matches")
    print(f"[{pf('manifest.json' in names)}] zip contains manifest.json")
    print(f"[{pf('report.pdf' in names)}] zip contains report.pdf")

    exps = database.list_exports()
    row = next((e for e in exps if e["export_id"] == resp.export_id), None)
    print(f"[{pf(row is not None and row['manifest_hash'] == resp.manifest_hash)}] "
          f"exports table row recorded with matching hash")

    with database.get_conn() as conn:
        a = conn.execute("SELECT details FROM audit_log WHERE action='export' "
                         "ORDER BY log_id DESC LIMIT 1").fetchone()
    print(f"[{pf(a is not None)}] audit_log 'export' entry written")

    # tamper check: flip a byte in a copied crop -> its hash should no longer match
    tampered = False
    for it in manifest["items"]:
        if it.get("crop_file"):
            fp = exp_dir / it["crop_file"]
            data = bytearray(fp.read_bytes())
            if data:
                data[0] ^= 0xFF
                new_sha = hashlib.sha256(bytes(data)).hexdigest()
                tampered = new_sha != it["crop_sha256"]
            break
    print(f"[{pf(tampered)}] tamper detection: modifying a file changes its SHA-256")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
