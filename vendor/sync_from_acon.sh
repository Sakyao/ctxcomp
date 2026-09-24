#!/usr/bin/env bash
#
# Re-copy the vendored harness from the upstream ACON checkout.
#
# Nothing in the suite reads that checkout at run time -- this script exists so the
# copy can be refreshed deliberately and diffed when upstream moves, and so the
# provenance in vendor/PROVENANCE.md can be regenerated rather than remembered.
#
#   bash vendor/sync_from_acon.sh
#   ACON=/path/to/other/checkout bash vendor/sync_from_acon.sh
#
# The mirrored trees are replaced wholesale (--delete): they are a copy, and a file
# that exists only in the copy is drift. Local changes meant to survive belong in the
# suite's own files or in patches/, not inside src/ or experiments/.
#
# After running this, update the digest in vendor/PROVENANCE.md:
#
#   find src experiments -type f -not -path "*__pycache__*" -not -name "*.pyc" \
#     | sort | xargs sha256sum | sha256sum
#
set -euo pipefail

ACON="${ACON:-${ACON_ROOT:-/z5s/morph/home/sjk/Agent/datasets/repos/acon}}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$ACON/src/productive_agents" ]; then
  echo "not an ACON checkout: $ACON" >&2
  exit 1
fi

echo "[sync] from $ACON"
echo "[sync] to   $HERE"

# Agent package: the compressors, the memory manager, the agent loop.
mkdir -p "$HERE/src"
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
  "$ACON/src/productive_agents" "$HERE/src/"

# Entry point and its assets. The excluded directories are run output, not code:
# outputs/ and experiments/ are written per round, data/ is a symlink to the
# AppWorld data root, shards/ and data_copy/ are generated splits.
mkdir -p "$HERE/experiments"
rsync -a --delete \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='outputs' --exclude='experiments' --exclude='shards' \
  --exclude='data' --exclude='data_copy' \
  "$ACON/experiments/appworld" "$HERE/experiments/"

# Metric implementation: analyze_experiment_tokens_v2, which defines Steps/Peak/Dep.
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
  "$ACON/experiments/analysis_tools" "$HERE/experiments/"

cp "$ACON/experiments/__init__.py" "$HERE/experiments/__init__.py"

# The two services the suite starts, and the endpoint-identity note.
mkdir -p "$HERE/experiments/repro"
for f in llmlingua_server.py fingerprint_endpoint.py embedding_server.py ENDPOINT_IDENTITY.md; do
  cp "$ACON/experiments/repro/$f" "$HERE/experiments/repro/$f"
done

echo "[sync] done. files: $(find "$HERE/src" "$HERE/experiments" -type f \
  -not -path '*__pycache__*' -not -name '*.pyc' | wc -l)"
echo "[sync] digest:"
find "$HERE/src" "$HERE/experiments" -type f -not -path '*__pycache__*' -not -name '*.pyc' \
  | sort | xargs sha256sum | sha256sum
