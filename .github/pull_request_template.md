<!-- Thanks for the PR — please complete each section so review is fast. -->

## Summary

<!-- One paragraph: what changed and why. -->

## Type of change

- [ ] Bug fix (no behaviour change to a passing test)
- [ ] Leakage / calibration fix (changes numbers in `outputs/*.json` — explain in "Headline impact" below)
- [ ] New feature / new baseline / new model
- [ ] Documentation only
- [ ] Refactor (no public-API change)

## Headline impact

<!-- If this PR changes a headline number reported in README.md or in any
     outputs/*.json artifact, paste the before/after numbers here. Numbers
     without a reproducible script will block the merge. -->

| Metric | Before | After |
|--------|--------|-------|
|        |        |       |

## Test plan

- [ ] `python -m pytest -q` passes locally
- [ ] `python -m ruff check src tests scripts` passes
- [ ] If v3 harness touched: `python scripts/v3_run.py` still produces a
      sensible `outputs/v3_metrics.json` (paste the verdict block)

## Leakage / honesty checklist

<!-- The project's distinguishing feature. Confirm each. -->

- [ ] No new full-series scaler fits — every scaler is fit on a train mask
- [ ] No look-ahead features (anything time-indexed uses past-only info at t)
- [ ] No `class_weight="balanced"` on rare-event labels without a calibration story
- [ ] No new `except Exception:` blocks that silently return NaN
- [ ] If a label was changed, the labeling function is still strictly causal

## Related issues

<!-- Closes #123 -->
