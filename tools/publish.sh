#!/usr/bin/env bash
# Veröffentlicht den aktuellen Stand auf GitHub Pages (Quelle main:/).
#   1) web/build_info.json  -> Zeitstempel + Notiz
#   2) web/sw.js            -> neuer Cache-Stamp (Browser holen SW-Skripte immer frisch,
#                              dadurch greift der neue Cache und alte Caches werden gelöscht)
#   3) commit + push        -> Pages baut automatisch
set -euo pipefail
cd "$(dirname "$0")/.."
NOTE="${1:-update}"
STAMP=$(date -u +%Y%m%d%H%M%S)
python3 - "$NOTE" "$STAMP" <<'PY'
import json, datetime, sys
note, stamp = sys.argv[1], sys.argv[2]
json.dump({
    "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "stamp": stamp,
    "note": note,
    "app": "CHORUS · FastVLM Live-Vision-Sprachchat",
    "pages": "https://lamachen.github.io/chorus/web/index.html",
}, open("web/build_info.json", "w"), ensure_ascii=False, indent=1)
PY
sed -i "s/const STAMP = '[^']*'/const STAMP = '$STAMP'/" web/sw.js
git add -A
if git diff --cached --quiet; then echo "nichts zu publizieren"; exit 0; fi
git commit -q -m "$NOTE"
git push -q origin main
echo "publiziert: $(git rev-parse --short HEAD) (stamp $STAMP) → https://lamachen.github.io/chorus/web/index.html"
