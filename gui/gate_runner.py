"""Чистая логика запуска гейта и парсинга отчёта (без streamlit) — тестируемо отдельно."""
from __future__ import annotations
import os, re, sys, glob, json, time, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

DATASETS = {
    "uci_banknote_auth": ("label", ""),
    "bank_churn": ("Exited", "Complain"),
    "creditcard_fraud_ulb": ("Class", "Time"),
    "german_credit": ("Risk", ""),
    "give_me_some_credit": ("SeriousDlqin2yrs", ""),
    "taiwan_credit_default": ("default.payment.next.month", ""),
}
VERSIONS = {
    "dataset_4": "✅ чистый эталон (d4)",
    "dataset_1": "☣️ label_flip (d1)",
    "dataset_2": "☣️ value-trigger / порча (d2)",
    "dataset_3": "☣️ порча признаков (d3)",
}
ST2DEC = {"passed": "ALLOW", "hand_check": "REVIEW", "failed": "BLOCK",
          "PASSED": "ALLOW", "HAND_CHECK": "REVIEW", "FAILED": "BLOCK"}
SEV = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}


def run_gate(ds: str, ver: str, target: str, drop: str, scope: str, ref: bool):
    """Вызвать реальный пайплайн через CLI, вернуть (report_dict | None, error | None, log)."""
    fs = sorted(glob.glob(str(ROOT / "data" / ds / f"{ver}.*")))
    if not fs:
        return None, f"нет файла data/{ds}/{ver}.*", ""
    cmd = [sys.executable, "-m", "src.cli", "scan", "--dataset", fs[0], "--target", target]
    if drop:
        cmd += ["--drop", drop]
    if scope == "Весь гейт (все слои)":
        cmd += ["--mode", "all"]
    else:
        cmd += ["--mode", "only", "--only", "model"]
    if ref and ver != "dataset_4":
        r4 = sorted(glob.glob(str(ROOT / "data" / ds / "dataset_4.*")))
        if r4:
            cmd += ["--reference", r4[0]]
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "таймаут скана (>10 мин)", ""
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    m = re.search(r"Report saved:\s*(\S+\.json)", out)
    path = m.group(1) if m else None
    if not path:
        cand = [f for f in glob.glob(str(REPORTS / "scan_report_*.json")) if os.path.getmtime(f) >= t0 - 1]
        path = max(cand, key=os.path.getmtime) if cand else None
    if not path or not os.path.exists(path):
        return None, "отчёт не найден (скан упал)", out
    return json.load(open(path)), None, out


def overall_decision(report: dict) -> str:
    decs = []
    for r in report.get("results", []):
        d = (r.get("details") or {}).get("decision")
        decs.append(str(d).upper() if d else ST2DEC.get(r.get("status", ""), "ALLOW"))
    return max(decs, key=lambda x: SEV.get(x, 0)) if decs else "ALLOW"
