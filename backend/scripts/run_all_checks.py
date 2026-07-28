"""Master verification sweep - runs every capability check in ONE process
(models load once) and prints a consolidated PASS/FAIL summary.

Order matters: the baseline re-ingest runs first (fresh 1336), then the feature
checks, then robustness (which snapshots/restores the baseline), then the API.

    python -u scripts/run_all_checks.py
"""
import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SECTIONS = [
    ("Baseline ingest + integrity", "reingest_and_verify"),
    ("License plate OCR", "verify_plates"),
    ("Multi-language translation", "verify_translate"),
    ("Forensic export (SHA-256)", "verify_forensics"),
    ("Robustness + unfamiliar footage", "verify_robustness"),
    ("REST API + WebSocket", "verify_api"),
]


def main():
    results = []
    for title, modname in SECTIONS:
        print(f"\n{'=' * 72}\n>>> {title}  ({modname}.py)\n{'=' * 72}", flush=True)
        buf = io.StringIO()
        error = None
        try:
            mod = __import__(modname)
            with redirect_stdout(buf):
                mod.main()
        except Exception as exc:  # noqa: BLE001
            error = exc
        out = buf.getvalue()
        print(out, flush=True)
        if error is not None:
            traceback.print_exc()
        n_pass = out.count("[PASS]")
        n_fail = out.count("[FAIL]") + (1 if error else 0)
        results.append((title, n_pass, n_fail, error))

    print(f"\n{'#' * 72}\n# FINAL VERIFICATION SUMMARY\n{'#' * 72}")
    total_pass = total_fail = 0
    for title, n_pass, n_fail, err in results:
        status = "OK  " if (n_fail == 0 and err is None) else "FAIL"
        extra = f"  ! {type(err).__name__}: {err}" if err else ""
        print(f"  [{status}] {title:36s} {n_pass:>2} pass / {n_fail} fail{extra}")
        total_pass += n_pass
        total_fail += n_fail
    print("#" * 72)
    print(f"  TOTAL: {total_pass} pass / {total_fail} fail across {len(results)} sections")
    print(f"  RESULT: {'ALL GREEN' if total_fail == 0 else 'SOME CHECKS FAILED'}")
    print("#" * 72)


if __name__ == "__main__":
    main()
