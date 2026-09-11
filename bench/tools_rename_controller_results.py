"""Prefix result files that carry a controller with '<controller>.' (harness naming fix, 2026-09-11). Idempotent."""
import glob, json, os
R = "/data/wookiekim/cgd/cgd-dev200/results/bench"
for jf in sorted(glob.glob(f"{R}/*/*@*.json")):
    base = os.path.basename(jf)
    recs = json.load(open(jf))
    ctrl = recs[0].get("controller", "") if recs else ""
    if not ctrl or base.startswith(ctrl + "."):
        continue
    new = os.path.join(os.path.dirname(jf), f"{ctrl}.{base}")
    os.rename(jf, new)
    csvf = jf.replace(".json", "_per_image.csv")
    if os.path.exists(csvf):
        os.rename(csvf, new.replace(".json", "_per_image.csv"))
    print("renamed", base, "->", os.path.basename(new))
print("done")
