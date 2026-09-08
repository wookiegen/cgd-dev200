#!/usr/bin/env python
"""04: Inject results/dev200_summary.md into README.md between the RESULTS markers. Idempotent."""
import os, re, datetime
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
readme = os.path.join(repo, "README.md"); summ = os.path.join(repo, "results", "dev200_summary.md")
body = open(summ).read().strip()
stamp = datetime.date.today().isoformat()
block = f"<!-- RESULTS:BEGIN -->\n_Last run: {stamp}. Regenerate with `bash scripts/run_all.sh`; this block is written by `scripts/04_fill_readme.py`._\n\n{body}\n<!-- RESULTS:END -->"
s = open(readme).read()
new = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda m: block, s, flags=re.S)
assert new != s or block in s, "RESULTS markers not found in README.md"
open(readme, "w").write(new); print("README results block updated from", summ)
