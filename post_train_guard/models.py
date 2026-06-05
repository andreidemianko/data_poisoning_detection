"""
Result model for the post_train_guard engine, mirroring dataset_guard/models.py:
a single decision (ALLOW / REVIEW / BLOCK) aggregated from per-detector findings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class FindingStatus(str, Enum):
    PASSED = "passed"     # детектор отработал, чисто
    REVIEW = "review"     # на ручную проверку (триаж)
    BLOCK = "block"       # явная атака -> блок
    SKIPPED = "skipped"   # неприменимо (нет модели/текста/метки и т.п.)
    ERROR = "error"       # ошибка выполнения детектора


# приоритет для сведения решения: чем выше severity, тем «хуже»
_SEVERITY = {
    FindingStatus.PASSED: 0,
    FindingStatus.SKIPPED: 0,
    FindingStatus.ERROR: 1,
    FindingStatus.REVIEW: 2,
    FindingStatus.BLOCK: 3,
}


@dataclass
class Finding:
    detector: str                       # имя детектора
    category: str                       # "text" | "tabular-data" | "tabular-model" | "nlp-model"
    status: FindingStatus
    verdict: str = ""                   # человекочитаемый итог
    details: Dict[str, Any] = field(default_factory=dict)
    advisory: bool = False              # True = триаж без опоры (не вердикт, не ведёт решение)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "category": self.category,
            "status": self.status.value,
            "verdict": self.verdict,
            "advisory": self.advisory,
            "details": self.details,
        }


@dataclass
class PostTrainReport:
    decision: Decision
    findings: List[Finding] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_findings(cls, findings: List[Finding], metadata: Dict[str, Any] | None = None
                      ) -> "PostTrainReport":
        """Свод вердикта с приоритетом КАЛИБРОВАННЫХ findings над триажем (advisory).

        1) любой BLOCK блокирует (безопасность);
        2) иначе, если есть калиброванные суждения (не advisory, вынесли PASSED/REVIEW
           со своей опорой — напр. model_diff при старой модели) — решает ХУДШИЙ
           среди них; advisory-REVIEW идут как приложенные строки, не вето;
        3) иначе (только триаж, режим 1) — как раньше: худший среди всех.
        """
        if any(f.status == FindingStatus.BLOCK for f in findings):
            worst = 3
        else:
            verdict = [f for f in findings if not f.advisory
                       and f.status in (FindingStatus.PASSED, FindingStatus.REVIEW)]
            if verdict:
                worst = max(_SEVERITY[f.status] for f in verdict)
            else:
                worst = max((_SEVERITY[f.status] for f in findings), default=0)
        decision = (Decision.BLOCK if worst == 3
                    else Decision.REVIEW if worst == 2
                    else Decision.ALLOW)
        return cls(decision=decision, findings=findings, metadata=metadata or {})

    def finding_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.status.value] = counts.get(f.status.value, 0) + 1
        return counts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "finding_counts": self.finding_counts(),
            "findings": [f.as_dict() for f in self.findings],
            "detectors": [f.detector for f in self.findings],
            "metadata": self.metadata,
        }
