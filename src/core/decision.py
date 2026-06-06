from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.scanners.base import ScanResult, ScanStatus, ScannerCategory


@dataclass
class FinalDecision:
    decision: str
    risk_score: float
    reasons: List[str] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    failed_scanners: List[str] = field(default_factory=list)
    review_scanners: List[str] = field(default_factory=list)
    skipped_scanners: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "risk_score": self.risk_score,
            "reasons": self.reasons,
            "category_scores": self.category_scores,
            "failed_scanners": self.failed_scanners,
            "review_scanners": self.review_scanners,
            "skipped_scanners": self.skipped_scanners,
        }


def decide(results: List[ScanResult]) -> FinalDecision:
    """
    Final aggregation policy.

    Logic:
    - hard security BLOCK from sanity/model => BLOCK;
    - stats failures add risk but do not automatically block unless many agree;
    - HAND_CHECK contributes to REVIEW;
    - scanner runtime errors contribute to REVIEW unless they are dataset/model load failures.
    """

    score = 0.0
    hard_block = False

    reasons: List[str] = []
    failed_scanners: List[str] = []
    review_scanners: List[str] = []
    skipped_scanners: List[str] = []

    category_scores = {
        "sanity": 0.0,
        "stats": 0.0,
        "model": 0.0,
        "runtime": 0.0,
    }

    for result in results:
        details = result.details or {}
        category = result.category.value

        if result.status == ScanStatus.SKIPPED:
            skipped_scanners.append(result.name)
            continue

        if result.status == ScanStatus.HAND_CHECK:
            review_scanners.append(result.name)

            if result.category == ScannerCategory.SANITY:
                add = 25.0
            elif result.category == ScannerCategory.MODEL:
                add = 25.0
            else:
                add = 8.0

            score += add
            category_scores[category] += add
            reasons.append(f"{result.name}: review")
            continue

        if result.status == ScanStatus.FAILED:
            failed_scanners.append(result.name)

            reason = str(details.get("reason", ""))

            # Pipeline load failures are real technical blockers.
            if reason in {"dataset_load_failed", "model_load_failed"}:
                hard_block = True
                score = max(score, 100.0)
                category_scores["runtime"] += 100.0
                reasons.append(f"{result.name}: technical load failure")
                continue

            # Dataset Guard / Post-Train Guard explicit BLOCK is a hard security block.
            explicit_decision = str(details.get("decision", "")).upper()

            if result.category in {ScannerCategory.SANITY, ScannerCategory.MODEL}:
                if explicit_decision == "BLOCK":
                    hard_block = True
                    score = max(score, 90.0)
                    category_scores[category] += 90.0
                    reasons.append(f"{result.name}: explicit BLOCK")
                else:
                    score += 35.0
                    category_scores[category] += 35.0
                    reasons.append(f"{result.name}: failed")
                continue

            # Stats are evidence, not always a hard block.
            if result.category == ScannerCategory.STATS:
                score += 12.0
                category_scores[category] += 12.0
                reasons.append(f"{result.name}: statistical anomaly")
                continue

            score += 20.0
            category_scores[category] += 20.0
            reasons.append(f"{result.name}: failed")

    # Agreement rule: many independent stats failures should block.
    stats_failed = [
        result for result in results
        if result.category == ScannerCategory.STATS and result.status == ScanStatus.FAILED
    ]
    stats_review = [
        result for result in results
        if result.category == ScannerCategory.STATS and result.status == ScanStatus.HAND_CHECK
    ]

    if len(stats_failed) >= 3:
        score += 20.0
        category_scores["stats"] += 20.0
        reasons.append("3+ stats scanners failed")

    if len(stats_failed) >= 2 and len(stats_review) >= 2:
        score += 15.0
        category_scores["stats"] += 15.0
        reasons.append("multiple stats scanners agree")

    score = min(100.0, score)

    if hard_block or score >= 70.0:
        decision = "BLOCK"
    elif score >= 25.0 or review_scanners or failed_scanners:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    return FinalDecision(
        decision=decision,
        risk_score=round(score / 100.0, 4),
        reasons=reasons[:30],
        category_scores={key: round(value / 100.0, 4) for key, value in category_scores.items()},
        failed_scanners=failed_scanners,
        review_scanners=review_scanners,
        skipped_scanners=skipped_scanners,
    )