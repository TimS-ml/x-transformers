#!/usr/bin/env python3
"""Prove an annotation pass changed only comments and docstrings, never behavior.

This fork carries a hand-written annotation layer on top of upstream. After an
upstream merge those annotations have to be reapplied, and it is easy to alter
real code by accident while doing it. This script parses the given revision's
version of each file and the working-tree version, strips all docstrings, and
compares the resulting ASTs. Comments never reach the AST at all, so any
difference means actual code changed.

Usage
-----
    scripts/check_comments_only.py <git-rev> [<file> ...]

With no files, every tracked *.py under x_transformers/ is checked. A file that
does not exist in <git-rev> (a new module arriving from upstream) is only
syntax-checked, and reported as NEW.

Examples
--------
    # after re-annotating on top of a merge commit
    scripts/check_comments_only.py HEAD

    # check the annotation commits that followed a specific merge
    scripts/check_comments_only.py <merge-sha> x_transformers/attend.py

Exit status is 0 when every file passes.
"""

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = "x_transformers"


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body.pop(0)
    return tree


def normalized(source):
    return ast.dump(strip_docstrings(ast.parse(source)))


def default_paths():
    package = ROOT / PACKAGE
    return sorted(
        str(path.relative_to(ROOT)) for path in package.glob("*.py")
    )


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    rev = argv[1]
    paths = argv[2:] or default_paths()
    failed = False

    for path in paths:
        rel = pathlib.Path(path)
        if rel.is_absolute():
            rel = rel.relative_to(ROOT)
        rel = rel.as_posix()

        shown = subprocess.run(
            ["git", "show", f"{rev}:{rel}"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

        working = (ROOT / rel).read_text()

        if shown.returncode != 0:
            print(f"NEW      {rel}  (absent from {rev}; syntax-check only)")
            try:
                ast.parse(working)
            except SyntaxError as err:
                print(f"         SYNTAX ERROR: {err}")
                failed = True
            continue

        try:
            before, after = normalized(shown.stdout), normalized(working)
        except SyntaxError as err:
            print(f"FAIL     {rel}  SYNTAX ERROR: {err}")
            failed = True
            continue

        if before == after:
            print(f"OK       {rel}")
        else:
            print(f"CHANGED  {rel}  <-- code semantics differ, not comments-only")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
