#!/usr/bin/env bash
# Starts the FastVLM video-call server with a self-signed certificate.
# HTTPS is mandatory: browsers only allow camera + microphone + speech
# recognition in a secure context, and "localhost" is the only insecure
# exception - a phone on your LAN is not.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8443}"
HOST="${HOST:-0.0.0.0}"
CERT_DIR="${CERT_DIR:-.certs}"
PY="${PYTHON:-python3}"

lan_ip() {
  if [[ "$(uname)" == "Darwin" ]]; then
    ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo ""
  else
    hostname -I 2>/dev/null | awk '{print $1}' || echo ""
  fi
}
IP="$(lan_ip)"

if ! $PY -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "→ Python-Pakete fehlen. Installiere:  $PY -m pip install -r requirements.txt"
  $PY -m pip install -r requirements.txt
fi

if [[ "${PLAIN_HTTP:-0}" == "1" ]]; then
  echo "⚠  Klartext-HTTP: Kamera/Mikro funktionieren damit nur auf localhost."
  exec $PY -m uvicorn server.main:app --host "$HOST" --port "$PORT" "$@"
fi

mkdir -p "$CERT_DIR"
if [[ ! -f "$CERT_DIR/cert.pem" ]]; then
  echo "→ Erzeuge selbstsigniertes Zertifikat (einmalig) …"
  SAN="DNS:localhost,IP:127.0.0.1"
  [[ -n "$IP" ]] && SAN="$SAN,IP:$IP"
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -subj "/CN=FastVLM Call" -addext "subjectAltName=$SAN" >/dev/null 2>&1
fi

echo
echo "  📞  FastVLM Video-Call läuft"
echo "      auf diesem Rechner :  https://localhost:$PORT"
[[ -n "$IP" ]] && echo "      vom Handy (WLAN)   :  https://$IP:$PORT"
echo
echo "      Handy & Rechner müssen im selben WLAN sein."
echo "      Beim ersten Öffnen warnt der Browser vor dem selbstsignierten"
echo "      Zertifikat:  Erweitert → „Trotzdem weiter“  (einmal pro Gerät)."
echo
echo "      Modell: ${FASTVLM_MODEL:-automatisch}   Backend: ${FASTVLM_BACKEND:-auto}"
echo "      Stoppen mit Strg+C"
echo

exec $PY -m uvicorn server.main:app --host "$HOST" --port "$PORT" \
  --ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem" \
  --ws-max-size 16777216 "$@"
