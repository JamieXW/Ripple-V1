"""Mine git history into change records — the free ground truth for impact eval.

Each non-merge commit is one labelled example: the set of files it touched is the
*real* blast radius of changing any one of them. We also map each commit's changed
lines back to the functions that contain them (parsing the file's blob *at that
commit*, so line numbers match), giving candidate seed functions for evaluation.

This miner is reused in M5 to generate training pairs (commit message, changed code).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import git

from ripple.parsing.parser import module_qname_from_path, parse_source

logger = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class CommitChange:
    """What one commit changed: the touched files and the functions within them."""

    sha: str
    changed_files: frozenset[str]  # repo-relative .py paths
    changed_functions: frozenset[str]  # qualified names whose body the commit edited


def _changed_new_lines(patch: str) -> set[int]:
    """Line numbers added/modified on the *new* side of a unified diff."""
    changed: set[int] = set()
    new_line = 0
    active = False
    for line in patch.splitlines():
        if line.startswith("@@"):
            match = _HUNK_RE.match(line)
            active = match is not None
            if match:
                new_line = int(match.group(1))
            continue
        if not active or line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            changed.add(new_line)
            new_line += 1
        elif line.startswith("-") or line.startswith("\\"):
            continue  # old-side line or "no newline" marker — doesn't advance new side
        else:
            new_line += 1  # context line
    return changed


def _blob_text(commit: git.Commit, path: str) -> str | None:
    """Read a file's contents as of ``commit`` (the new side of the diff)."""
    try:
        blob = commit.tree / path
    except KeyError:
        return None
    try:
        text: str = blob.data_stream.read().decode("utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        logger.debug("could not read %s@%s: %s", path, commit.hexsha[:8], exc)
        return None
    return text


def _intersects(lines: set[int], start: int, end: int) -> bool:
    return any(start <= ln <= end for ln in lines)


def mine_commits(
    repo_path: Path,
    max_count: int = 300,
    min_files: int = 2,
    max_files: int = 10,
) -> list[CommitChange]:
    """Walk up to ``max_count`` commits, keeping focused ones (``min..max`` .py files)."""
    repo_root = Path(repo_path)
    repo = git.Repo(repo_root)
    changes: list[CommitChange] = []

    for commit in repo.iter_commits("HEAD", max_count=max_count):
        if len(commit.parents) != 1:  # skip merges and the root commit
            continue
        parent = commit.parents[0]
        py_diffs = [
            d
            for d in parent.diff(commit, create_patch=True)
            if d.b_path and d.b_path.endswith(".py") and not d.deleted_file
        ]
        if not (min_files <= len(py_diffs) <= max_files):
            continue

        changed_files: set[str] = set()
        changed_functions: set[str] = set()
        for diff in py_diffs:
            path = diff.b_path
            assert path is not None  # guaranteed by the filter above
            changed_files.add(path)
            raw = diff.diff
            patch = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else (raw or "")
            lines = _changed_new_lines(patch)
            content = _blob_text(commit, path)
            if not lines or content is None:
                continue
            module_qname = module_qname_from_path(repo_root / path, repo_root)
            try:
                parsed = parse_source(content, module_qname, path)
            except SyntaxError:
                continue
            changed_functions.update(
                node.qualified_name
                for node in parsed.nodes
                if node.kind == "function" and _intersects(lines, node.start_line, node.end_line)
            )

        changes.append(
            CommitChange(
                sha=commit.hexsha,
                changed_files=frozenset(changed_files),
                changed_functions=frozenset(changed_functions),
            )
        )
    return changes
