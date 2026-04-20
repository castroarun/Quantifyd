#!/usr/bin/env bash
# Backup backtest_data/ to a GitHub release asset.
#
# Usage:
#   export GITHUB_TOKEN=ghp_...          # required, needs repo scope
#   export GITHUB_REPO=castroarun/Quantifyd   # optional, autodetected from git remote
#   scripts/backup_to_github_release.sh             # daily tag (backup-YYYYMMDD)
#   scripts/backup_to_github_release.sh manual      # custom tag suffix
#
# Creates a tarball excluding secrets and rebuildable caches, then uploads it
# as a release asset. Release assets are capped at 2 GB each by GitHub — if
# the tarball exceeds that, the script will split it into parts.

set -euo pipefail

# ─── Resolve repo root ───────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d backtest_data ]]; then
  echo "Error: backtest_data/ not found at $REPO_ROOT" >&2
  exit 1
fi

# ─── Config ──────────────────────────────────────────────────
: "${GITHUB_TOKEN:?GITHUB_TOKEN env var is required}"

if [[ -z "${GITHUB_REPO:-}" ]]; then
  # Autodetect from git remote
  remote_url="$(git remote get-url origin 2>/dev/null || true)"
  if [[ "$remote_url" =~ github\.com[:/]([^/]+/[^/\.]+) ]]; then
    GITHUB_REPO="${BASH_REMATCH[1]}"
  else
    echo "Error: cannot autodetect GITHUB_REPO from git remote. Export it." >&2
    exit 1
  fi
fi

DATE="$(date +%Y%m%d_%H%M%S)"
TAG_SUFFIX="${1:-$DATE}"
TAG="backup-${TAG_SUFFIX}"
ARCHIVE_NAME="backtest_data_${TAG_SUFFIX}.tar.gz"
TMP_ARCHIVE="/tmp/${ARCHIVE_NAME}"
MAX_ASSET_BYTES=$((2 * 1024 * 1024 * 1024 - 100 * 1024 * 1024))  # 1.9 GB (stay under 2GB ceiling)

echo "Repo:    $GITHUB_REPO"
echo "Tag:     $TAG"
echo "Archive: $TMP_ARCHIVE"

# ─── Create tarball ──────────────────────────────────────────
echo "Creating tarball (this may take a minute)..."
tar --exclude='backtest_data/access_token.json' \
    --exclude='backtest_data/fundamentals_cache' \
    --exclude='backtest_data/__pycache__' \
    -czf "$TMP_ARCHIVE" backtest_data/

ARCHIVE_BYTES=$(wc -c < "$TMP_ARCHIVE")
ARCHIVE_SIZE_H=$(du -h "$TMP_ARCHIVE" | cut -f1)
echo "Archive size: $ARCHIVE_SIZE_H ($ARCHIVE_BYTES bytes)"

# ─── Split if needed ─────────────────────────────────────────
UPLOAD_FILES=("$TMP_ARCHIVE")
if (( ARCHIVE_BYTES > MAX_ASSET_BYTES )); then
  echo "Archive exceeds 1.9 GB — splitting into parts..."
  split -b 1800M -d --additional-suffix=.part "$TMP_ARCHIVE" "${TMP_ARCHIVE}."
  UPLOAD_FILES=()
  for f in "${TMP_ARCHIVE}."*.part; do UPLOAD_FILES+=("$f"); done
  rm "$TMP_ARCHIVE"
  echo "Split into ${#UPLOAD_FILES[@]} parts. To restore:"
  echo "  cat ${ARCHIVE_NAME}.*.part > ${ARCHIVE_NAME} && tar xzf ${ARCHIVE_NAME}"
fi

# ─── Create release ──────────────────────────────────────────
API_BASE="https://api.github.com/repos/${GITHUB_REPO}"
AUTH_HEADER="Authorization: token ${GITHUB_TOKEN}"

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo "")"
NOTES="Automated backtest_data/ backup.

- Created: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Commit: ${COMMIT_SHA}
- Host: $(hostname 2>/dev/null || echo unknown)
- Size: ${ARCHIVE_SIZE_H}
- Excludes: access_token.json, fundamentals_cache/

To restore:
  curl -L -H \"Authorization: token \$GITHUB_TOKEN\" -o ${ARCHIVE_NAME} \\
    https://github.com/${GITHUB_REPO}/releases/download/${TAG}/${ARCHIVE_NAME}
  tar xzf ${ARCHIVE_NAME} -C /path/to/restore/"

echo "Creating release $TAG..."
release_json=$(curl --ssl-no-revoke -fsS -H "$AUTH_HEADER" -H "Accept: application/vnd.github+json" \
  -X POST "${API_BASE}/releases" \
  -d "$(python3 -c "
import json, os, sys
print(json.dumps({
  'tag_name': '$TAG',
  'name': 'Backup ${TAG_SUFFIX}',
  'body': '''$NOTES''',
  'draft': False,
  'prerelease': True,
}))
")")

UPLOAD_URL=$(echo "$release_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['upload_url'].split('{')[0])")
RELEASE_URL=$(echo "$release_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['html_url'])")

echo "Release created: $RELEASE_URL"
echo "Uploading asset(s)..."

for f in "${UPLOAD_FILES[@]}"; do
  name="$(basename "$f")"
  echo "  uploading $name..."
  curl --ssl-no-revoke -fsS -H "$AUTH_HEADER" \
    -H "Content-Type: application/octet-stream" \
    --data-binary "@${f}" \
    "${UPLOAD_URL}?name=${name}" > /dev/null
done

# Cleanup
rm -f "${TMP_ARCHIVE}" "${TMP_ARCHIVE}."*.part 2>/dev/null || true

echo ""
echo "Done. Release: $RELEASE_URL"
