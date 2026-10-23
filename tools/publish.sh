#!/usr/bin/env bash
# Veröffentlicht den aktuellen Stand: build_info schreiben, committen, pushen.
# GitHub Pages (Quelle main:/) baut danach automatisch -> https://lamachen.github.io/chorus/
set -euo pipefail
cd "$(dirname "$0")/.."
NOTE="${1:-update}"
python3 - "$NOTE" <<'PY'
import json, datetime, sys
json.dump({
    "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "note": sys.argv[1],
    "app": "CHORUS · FastVLM Live-Vision-Sprachchat",
    "pages": "https://lamachen.github.io/chorus/web/index.html",
}, open("web/build_info.json", "w"), ensure_ascii=False, indent=1)
PY
git add -A
if git diff --cached --quiet; then echo "nichts zu publizieren"; exit 0; fi
git commit -q -m "$NOTE"
git push -q origin main
echo "publiziert: $(git rev-parse --short HEAD)  →  https://lamachen.github.io/chorus/web/index.html"
