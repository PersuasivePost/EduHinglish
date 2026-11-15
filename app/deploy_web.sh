#!/usr/bin/env bash
# Builds the Flutter web app against the Render backend and publishes build/web to Vercel.
# Usage (from app/):  ./deploy_web.sh            -> production deploy
set -euo pipefail
BACKEND_URL="${BACKEND_URL:-https://eduhinglish.onrender.com}"
# SKIP_BUILD=1 reuses the existing build/web (e.g. after fixing a Vercel login problem)
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  flutter build web --release --dart-define=BACKEND_URL="$BACKEND_URL"
fi
cp vercel.json build/web/vercel.json
cd build/web
npx --yes vercel@latest deploy --prod
