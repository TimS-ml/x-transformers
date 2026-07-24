# scripts

Two guards for keeping this fork's annotation layer and docs honest across
upstream merges. Both are stdlib-only, take no arguments in the common case,
and can run from any working directory.

## After an upstream merge

```bash
git merge upstream/main            # resolve, commit
scripts/check_comments_only.py HEAD   # after re-annotating
scripts/check_doc_symbols.py
```

## `check_comments_only.py`

Proves an annotation pass changed only comments and docstrings. It strips
docstrings from both the given revision and the working tree, then compares
ASTs — comments never reach the AST, so any difference is real code drift.

Run it with the merge commit as the revision, after reapplying annotations:

```bash
scripts/check_comments_only.py HEAD
scripts/check_comments_only.py <merge-sha> x_transformers/attend.py
```

New modules arriving from upstream are absent from the revision, so they are
syntax-checked and reported as `NEW` rather than compared.

## `check_doc_symbols.py`

Verifies every symbol anchor in the docs still resolves against the AST of
`x_transformers/`.

The docs previously referenced source by `file:line`. Those anchors rotted on
every merge and every annotation pass, because both shift everything below
them — one sync left roughly 95% of them wrong, drifting by up to 186 lines.
They are now symbol anchors:

```
`xt::AttentionLayers.forward`     `attend::Attend`
`xl::TTTModuleWrapper`            `lejepa::LatentAutoregressive.sigreg_loss`
```

The text after `::` is an exact Python dotted path: a class, a method, or a
module-level function. Constructor keyword arguments and plain attributes are
not symbols — reference the owning class instead:

```
sparse_topk (`xt::Attention`)          correct
`xt::Attention.sparse_topk`            wrong, will not resolve
```

The abbreviation table lives at the top of the script and matches the legend
each doc prints in its own introduction. Paths may be absolute, so notes kept
outside the repo can be checked with the same command.
