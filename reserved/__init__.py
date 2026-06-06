"""
RESERVED — detectors parked for later integration into other modules.

Nothing here is wired into the pipeline or into post_train_guard. These are the
data-level / consensus detectors removed from the active post-train ensemble:
charset (homoglyph), backdoor trigger tokens, SecureLearn, kNN consensus
(tabular and on BERT embeddings).

When a target module exists (e.g. a data-level guard), import the relevant
checks from reserved.checks and add them to that module's registry.
"""
