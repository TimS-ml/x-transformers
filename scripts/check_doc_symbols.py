#!/usr/bin/env python3
"""Verify every symbol anchor in a markdown doc still resolves.

The docs under docs/ used to reference source by `file:line`. Those anchors
rotted on every upstream merge and every annotation pass, because both shift
everything below them - a sync once left roughly 95% of them wrong, drifting by
up to 186 lines. They were replaced with symbol anchors:

    `xt::AttentionLayers.forward`      `attend::Attend`
    `xl::TTTModuleWrapper`             `lejepa::LatentAutoregressive.sigreg_loss`

The text after `::` is an exact Python dotted path - a class, a method, or a
module-level function. Constructor keyword arguments and plain attributes are
not symbols and must not be written as anchors; reference the owning class
instead, as in ``sparse_topk (`xt::Attention`)``.

This script re-derives every symbol from the AST of x_transformers/ and reports
any anchor that no longer resolves. Run it after an upstream merge.

Usage
-----
    scripts/check_doc_symbols.py [<markdown-file> ...]

With no arguments, every *.md under docs/ is checked. Paths may be absolute, so
notes kept outside the repo can be checked too.

Exit status is 0 when every anchor resolves.
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "x_transformers"
DOCS = ROOT / "docs"

# doc abbreviation -> module filename. Keep in step with the legend each doc
# prints in its own introduction.
ABBREV = {
    "xt": "x_transformers.py",
    "attend": "attend.py",
    "arw": "autoregressive_wrapper.py",
    "xl": "xl_autoregressive_wrapper.py",
    "nar": "nonautoregressive_wrapper.py",
    "bsw": "belief_state_wrapper.py",
    "cont": "continuous.py",
    "cae": "continuous_autoencoder.py",
    "ebt": "entropy_based_tokenizer.py",
    "ft": "free_transformer.py",
    "lejepa": "gpt_lejepa.py",
    "vae": "gpt_vae.py",
    "mi": "multi_input.py",
    "nmlp": "neo_mlp.py",
    "nlw": "next_latent_wrapper.py",
    "up": "up_wrapper.py",
    "xval": "xval.py",
    "dpo": "dpo.py",
}

ANCHOR = re.compile(r"`([a-z_]+)::([A-Za-z_][A-Za-z0-9_.]*)`")


def symbols_of(filename):
    """Every class, method and module-level function defined in a module."""
    path = PACKAGE / filename
    if not path.exists():
        return None

    found = set()

    def walk(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                found.add(name)
                walk(child, name + ".")
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        found.add(prefix + target.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                found.add(prefix + child.target.id)

    walk(ast.parse(path.read_text()))
    return found


def main(argv):
    docs = argv[1:] or sorted(str(path) for path in DOCS.glob("*.md"))
    if not docs:
        print(f"no markdown files found under {DOCS}")
        return 2

    cache, unresolved, total = {}, [], 0

    for doc in docs:
        text = pathlib.Path(doc).read_text()
        label = pathlib.Path(doc).name

        for abbrev, symbol in ANCHOR.findall(text):
            total += 1

            if abbrev not in ABBREV:
                unresolved.append((label, f"{abbrev}::{symbol}", f"unknown abbreviation '{abbrev}'"))
                continue

            filename = ABBREV[abbrev]
            if filename not in cache:
                cache[filename] = symbols_of(filename)

            known = cache[filename]
            if known is None:
                unresolved.append((label, f"{abbrev}::{symbol}", f"module {filename} no longer exists"))
            elif symbol not in known:
                unresolved.append((label, f"{abbrev}::{symbol}", f"not defined in {filename}"))

    print(f"checked {total} anchors across {len(docs)} doc(s)")

    if not unresolved:
        print("ALL ANCHORS RESOLVE")
        return 0

    print(f"\n{len(unresolved)} UNRESOLVED:")
    for label, anchor, reason in unresolved:
        print(f"  {label}: `{anchor}`  -- {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
