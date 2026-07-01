# Usage

EMMA provides three practical workflows:

1. `emma restore` with a user-provided mask.
2. `emma restore --auto-mask` for bin-level missing-region detection followed by restoration.
3. `emma reconstruct` for conservative matrix reconstruction without an explicit missing mask.

For real missing-region restoration, prefer an explicit BED or bin-region file when you know the failed genomic bins. Use `--auto-mask` when the failed bins are not known in advance.
