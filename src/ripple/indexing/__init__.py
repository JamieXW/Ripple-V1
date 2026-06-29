"""Indexing (M1 full, M7 incremental): orchestrate parse -> graph + embed -> store.
Incremental reindexing re-processes only changed files + affected dependents,
detected via per-file content hashes (Merkle approach)."""
