# reserved/ — parked detectors

Detectors removed from the active **post_train_guard** ensemble, kept here until
they are integrated into other modules. Not imported by the pipeline.

| detector | kind | future home (idea) |
|---|---|---|
| charset / homoglyph | text, rule | data-level guard |
| backdoor trigger tokens | text, rule | data-level guard |
| SecureLearn | tabular data-level | data-level guard |
| kNN label consensus | tabular data-level | data-level guard |
| kNN on BERT embeddings | NLP, model-level | NLP data/label guard |

- Logic: `reserved/detectors.py`
- Ready-to-use checks: `reserved/checks.py` (`RESERVED_CHECKS`)

**To revive:** import the check from `reserved.checks` and add it to the target
module's `CHECKS` list. The checks already return the same `Finding` objects.
