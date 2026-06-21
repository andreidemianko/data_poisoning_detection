import traceback
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.core.loaders import load_dataset, load_model, project_root
from src.core.report import ReportWriter, build_report, ScanReport
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory, ScanStatus


@dataclass
class ExecutionPlan:
    categories: Sequence[ScannerCategory]
    continue_on_fail: bool = False
    model_requires_previous_pass: bool = False


class SecurityPipeline:
    def __init__(self, scanners: Iterable[BaseScanner]):
        self.scanners = list(scanners)

    def _filter_scanners(self, categories: Sequence[ScannerCategory]) -> List[BaseScanner]:
        return [scanner for scanner in self.scanners if scanner.category in categories]

    @staticmethod
    def _safe_run_scanner(scanner: BaseScanner, context: ScanContext) -> ScanResult:
        """Run scanner safely. Scanner crash becomes a ScanResult, not a Python traceback."""
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                result = scanner.run(context)

                if not isinstance(result, ScanResult):
                    return ScanResult(
                        name=scanner.name,
                        category=scanner.category,
                        status=ScanStatus.FAILED,
                        passed=False,
                        details={
                            "reason": "scanner_invalid_result",
                            "error": f"Scanner returned {type(result).__name__}, expected ScanResult.",
                        },
                    )

                if caught:
                    details = dict(result.details or {})
                    details["warnings"] = [
                        {
                            "category": warning.category.__name__,
                            "message": str(warning.message),
                        }
                        for warning in caught[:20]
                    ]

                    result = ScanResult(
                        name=result.name,
                        category=result.category,
                        status=result.status,
                        passed=result.passed,
                        details=details,
                    )

                return result

        except Exception as exc:
            return ScanResult(
                name=scanner.name,
                category=scanner.category,
                status=ScanStatus.FAILED,
                passed=False,
                details={
                    "reason": "scanner_runtime_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc()[-3000:],
                },
            )

    def execute(
        self,
        dataset_path: str,
        model_path: str,
        plan: ExecutionPlan,
        reports_dir: Optional[Path] = None,
    ) -> ScanReport:
        run_id = uuid.uuid4().hex[:8]
        results: List[ScanResult] = []
        report_writer = ReportWriter(reports_dir or project_root() / "reports")

        try:
            dataset_bundle = load_dataset(dataset_path)
        except Exception as exc:
            result = ScanResult(
                name="Pipeline: load dataset",
                category=ScannerCategory.SANITY,
                status=ScanStatus.FAILED,
                passed=False,
                details={
                    "reason": "dataset_load_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "dataset_path": dataset_path,
                },
            )
            results.append(result)
            report = build_report(run_id, dataset_path, model_path, results)
            report_path = report_writer.write(report)
            print("❌ Failed")
            print(f"🧾 Report saved: {report_path}")
            return report

        context = ScanContext(
            dataset_path=dataset_bundle.path,
            model_path=model_path,
            dataset=dataset_bundle.data,
        )

        model_loaded = False

        scanners = self._filter_scanners(plan.categories)
        scanners_by_category = {
            category: [scanner for scanner in scanners if scanner.category == category]
            for category in plan.categories
        }

        for category in plan.categories:
            category_failed = False

            for scanner in scanners_by_category.get(category, []):
                print(f"⏳ Running: {scanner.name} ({scanner.category.value})")

                if scanner.category == ScannerCategory.MODEL and not model_loaded:
                    try:
                        model_bundle = load_model(model_path)
                    except Exception as exc:
                        result = ScanResult(
                            name="Pipeline: load model",
                            category=ScannerCategory.MODEL,
                            status=ScanStatus.FAILED,
                            passed=False,
                            details={
                                "reason": "model_load_failed",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "model_path": model_path,
                            },
                        )
                        results.append(result)
                        self._print_result_status(result)
                        category_failed = True
                        break

                    context.model_path = model_bundle.path
                    context.model_state = model_bundle.state_dict
                    context.metadata.update(model_bundle.metadata)
                    model_loaded = True

                result = self._safe_run_scanner(scanner, context)
                results.append(result)

                self._print_result_status(result)

                if result.status == ScanStatus.FAILED:
                    category_failed = True

            if category_failed:
                if plan.model_requires_previous_pass and category in {
                    ScannerCategory.SANITY,
                    ScannerCategory.STATS,
                }:
                    break

                if not plan.continue_on_fail:
                    break

        report = build_report(run_id, context.dataset_path, context.model_path, results)
        report_path = report_writer.write(report)
        print(f"🧾 Report saved: {report_path}")

        final = report.metadata.get("final_decision", {})
        if final:
            print(f"🏁 Final decision: {final.get('decision')} | risk={final.get('risk_score')}")
            for reason in final.get("reasons", [])[:5]:
                print(f"   - {reason}")

        return report

    @staticmethod
    def _print_result_status(result: ScanResult) -> None:
        if result.status == ScanStatus.PASSED:
            print("✅ Passed")
            return

        if result.status == ScanStatus.SKIPPED:
            print("⏭️  Skipped")
            return

        if result.status == ScanStatus.HAND_CHECK:
            print("⚠️  Review")
            return

        print("❌ Failed")
