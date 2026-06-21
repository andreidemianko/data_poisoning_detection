import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.scanners.base import ScanResult, ScanStatus
from src.core.loaders import project_root
from src.core.decision import decide


@dataclass
class ScanReport:
    run_id: str
    timestamp: str
    dataset_path: str
    model_path: str
    results: List[ScanResult]
    overall_status: ScanStatus
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "model_path": self.model_path,
            "overall_status": self.overall_status.value,
            "results": [
                {
                    "name": result.name,
                    "category": result.category.value,
                    "status": result.status.value,
                    "passed": result.passed,
                    "details": result.details,
                }
                for result in self.results
            ],
            "metadata": self.metadata,
        }


class ReportWriter:
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir

    def write(self, report: ScanReport) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"scan_report_{report.run_id}.json"
        path = self.reports_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        return path


def build_report(run_id: str, dataset_path: str, model_path: str, results: List[ScanResult]) -> ScanReport:
    root = project_root()
    dataset_path_obj = Path(dataset_path).resolve()
    model_path_obj = Path(model_path).resolve()
    try:
        dataset_rel = str(dataset_path_obj.relative_to(root))
        model_rel = str(model_path_obj.relative_to(root))
    except ValueError:
        dataset_rel = str(dataset_path_obj)
        model_rel = str(model_path_obj)

    final = decide(results)

    if final.decision == "BLOCK":
        overall_status = ScanStatus.FAILED
    elif final.decision == "REVIEW":
        overall_status = ScanStatus.HAND_CHECK
    else:
        overall_status = ScanStatus.PASSED

    return ScanReport(
        run_id=run_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        dataset_path=dataset_rel,
        model_path=model_rel,
        results=results,
        overall_status=overall_status,
        metadata={
            "final_decision": final.as_dict(),
        },
    )
