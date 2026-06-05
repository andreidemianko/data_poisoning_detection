"""
RESERVED checks (data-level + consensus). NOT part of any active ensemble.
They reuse the shared helpers/result-factories from post_train_guard so they can
be lifted into a future module (e.g. data_guard) with minimal changes.

To revive one: import it and add to that module's registry CHECKS.
"""
from __future__ import annotations

from typing import Callable, List

from post_train_guard.detectors import common as cm
from post_train_guard.detectors import nlp_model_level as nlp
from post_train_guard.models import Finding, FindingStatus
from post_train_guard.registry import ScanInput, err, from_scores, skip

from reserved import detectors as rd


def check_charset(inp: ScanInput) -> Finding:
    n, c = "charset / homoglyph", "text"
    cols = cm.text_columns(inp.dataset)
    if not cols:
        return skip(n, c, "no free-text columns")
    texts = cm.combined_text(inp.dataset, cols)
    rows, examples = rd.charset_scan(texts)
    det = {"text_columns": cols, "n_rows": len(texts), "homoglyph_rows": rows, "examples": examples}
    block_rows = inp.config.get("charset_block_rows", 2)
    block_frac = inp.config.get("charset_block_frac", 0.001)
    if rows >= max(block_rows, block_frac * len(texts)):
        return Finding(n, c, FindingStatus.BLOCK, f"mixed-script words in {rows} rows", det)
    return Finding(n, c, FindingStatus.PASSED, "no homoglyphs", det)


def check_rare_token(inp: ScanInput) -> Finding:
    n, c = "backdoor trigger tokens", "text"
    cols = cm.text_columns(inp.dataset)
    if not cols:
        return skip(n, c, "no free-text columns")
    texts = cm.combined_text(inp.dataset, cols)
    rows, tokens = rd.trigger_scan(texts)
    det = {"text_columns": cols, "n_rows": len(texts), "trigger_rows": rows, "trigger_tokens": tokens[:20]}
    if tokens:
        return Finding(n, c, FindingStatus.BLOCK, f"trigger token(s) {tokens[:5]} in {rows} rows", det)
    return Finding(n, c, FindingStatus.PASSED, "no injected trigger tokens", det)


def check_securelearn(inp: ScanInput) -> Finding:
    n, c = "SecureLearn (per-class feature outliers)", "tabular-data"
    try:
        X, y, label_col = cm.extract_xy(inp.dataset)
    except ValueError as exc:
        return skip(n, c, str(exc))
    if len(X) < inp.config["min_rows"]:
        return skip(n, c, f"too few rows ({len(X)})")
    s = rd.securelearn_scores(X, y)
    return from_scores(n, c, s, inp.config, "feature-outlier rows",
                       "no feature-outlier subpopulation", {"label_column": label_col, "n_rows": len(X)})


def check_knn(inp: ScanInput) -> Finding:
    n, c = "kNN label consensus", "tabular-data"
    try:
        X, y, label_col = cm.extract_xy(inp.dataset)
    except ValueError as exc:
        return skip(n, c, str(exc))
    if len(X) < inp.config["min_rows"]:
        return skip(n, c, f"too few rows ({len(X)})")
    s = rd.knn_consensus_scores(X, y)
    return from_scores(n, c, s, inp.config, "rows disagree with neighbours",
                       "labels consistent with neighbours", {"label_column": label_col, "n_rows": len(X)})


def check_nlp_knn(inp: ScanInput) -> Finding:
    n, c = "kNN consensus on BERT embeddings", "nlp-model"
    try:
        emb, y, label_col, tcols = nlp.representation(inp.model_path, inp.dataset)
    except ValueError as exc:
        return skip(n, c, str(exc))
    except ImportError:
        return skip(n, c, "transformers/torch not installed: pip install transformers")
    except Exception as exc:  # noqa: BLE001
        return err(n, c, "BERT encode failed", exc)
    if len(emb) < inp.config["min_rows"]:
        return skip(n, c, f"too few rows ({len(emb)})")
    s = rd.knn_consensus_scores(emb, y)
    return from_scores(n, c, s, inp.config, "rows disagree with BERT neighbours",
                       "labels consistent with BERT neighbours",
                       {"text_columns": tcols, "label_column": label_col, "n_rows": len(emb)})


# Зарезервированные проверки — НЕ подключены ни к одному gate.
RESERVED_CHECKS: List[Callable[[ScanInput], Finding]] = [
    check_charset,        # text  (homoglyph)
    check_rare_token,     # text  (backdoor trigger)
    check_securelearn,    # tabular data-level (feature outliers)
    check_knn,            # tabular data-level (label flip)
    check_nlp_knn,        # NLP, on BERT embeddings (label flip)
]
