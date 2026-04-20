#!/usr/bin/env bash
# Restore backtest_data/ from a GitHub release asset created by
# scripts/backup_to_github_release.sh.
#
# Usage:
#   export GITHUB_TOKEN=ghp_...
#   scripts/restore_from_github_release.sh                  # latest backup-* release
#   scripts/restore_from_github_release.sh backup-20260420  # specific tag
#
# Restores into backtest_data/. Will NOT overwrite unless --force passed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${GITHUB_TOKEN:?GITHUB_TOKEN env var is required}"

if [[ -z "${GITHUB_REPO:-}" ]]; then
  remote_url="$(git remote get-url origin 2>/dev/null || true)"
  if [[ "$remote_url" =~ github\.com[:/]([^/]+/[^/\.]+) ]]; then
    GITHUB_REPO="${BASH_REMATCH[1]}"
  else
    echo "Error: cannot autodetect GITHUB_REPO" >&2
    exit 1
  fi
fi

TAG="${1:-}"
FORCE="${2:-}"

API_BASE="https://api.github.com/repos/${GITHUB_REPO}"
AUTH_HEADER="Authorization: token ${GITHUB_TOKEN}"

# ─── Find release ────────────────────────────────────────────
if [[ -z "$TAG" ]]; then
  echo "Finding latest backup-* release..."
  TAG=$(curl --ssl-no-revoke -fsS -H "$AUTH_HEADER" "${API_BASE}/releases?per_page=30" | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data:
    if r['tag_name'].startswith('backup-'):
        print(r['tag_name']); break
")
  if [[ -z "$TAG" ]]; then
    echo "Error: no backup-* release found" >&2
    exit 1
  fi
  echo "Latest: $TAG"
fi

# ─── Fetch asset list ────────────────────────────────────────
release_json=$(curl --ssl-no-revoke -fsS -H "$AUTH_HEADER" "${API_BASE}/releases/tags/${TAG}")
mapfile -t ASSETS < <(echo "$release_json" | python3 -c "
import json, sys
r = json.load(sys.stdin)
for a in r.get('assets', []):
    print(f\"{a['id']}\\t{a['name']}\")
")

if [[ ${#ASSETS[@]} -eq 0 ]]; then
  echo "Error: no assets on release $TAG" >&2
  exit 1
fi

# ─── Safety: prompt unless --force ───────────────────────────
if [[ -d backtest_data && "$FORCE" != "--force" ]]; then
  echo "WARNING: backtest_data/ already exists. Files may be overwritten."
  read -r -p "Continue? [y/N] " ans
  [[ "$ans" =~ ^[yY]$ ]] || { echo "Aborted."; exit 0; }
fi

# ─── Download ────────────────────────────────────────────────
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for entry in "${ASSETS[@]}"; do
  id="${entry%%	*}"
  name="${entry##*	}"
  echo "Downloading $name..."
  curl --ssl-no-revoke -fL -H "$AUTH_HEADER" -H "Accept: application/octet-stream" \
    -o "${TMP_DIR}/${name}" \
    "${API_BASE}/releases/assets/${id}"
done

# ─── Reassemble split parts if present ───────────────────────
cd "$TMP_DIR"
if ls *.part >/dev/null 2>&1; then
  base=$(ls *.part | head -1 | sed 's/\.[0-9]*\.part$//')
  echo "Reassembling split archive into $base..."
  cat "$base".*.part > "$base"
  rm "$base".*.part
fi

# ─── Extract ─────────────────────────────────────────────────
archive=$(ls *.tar.gz | head -1)
echo "Extracting $archive into $REPO_ROOT..."
tar xzf "$archive" -C "$REPO_ROOT"

echo ""
echo "Restore complete from $TAG."
