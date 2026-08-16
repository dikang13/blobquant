#!/usr/bin/env bash
# Download the model weights from the GitHub release into the repository root.
#
# The weights are 261 MB, which is over GitHub's 100 MB per-file limit for ordinary
# git, so they are attached to a release rather than committed. Release assets have
# no bandwidth quota, unlike Git LFS.
#
#   ./scripts/fetch_weights.sh              # into the repository root
#   ./scripts/fetch_weights.sh /some/dir    # somewhere else
#
# Re-running is cheap: an existing file with the right checksum is left alone.
#
# Maintainer: to publish a new copy, strip the training checkpoint and attach it.
#
#   uv run python scripts/strip_checkpoint.py model_best_checkpoint.pytorch model_weights.pytorch
#   gh release create weights-v1 model_weights.pytorch \
#       --repo dikang13/blobquant --title "Model weights v1" --notes "..."
#
# Then update SHA256 below to match `sha256sum model_weights.pytorch`.

set -euo pipefail

REPO="${BLOBQUANT_REPO:-dikang13/blobquant}"
TAG="${BLOBQUANT_WEIGHTS_TAG:-weights-v1}"
ASSET="model_weights.pytorch"
SHA256="75d947fdae6cfe7790d129b2b2b855cc55ac8cf3c3b5570adcb730c63964afff"

destdir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
target="${destdir%/}/${ASSET}"
url="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

verify() {
    [ -f "$1" ] || return 1
    echo "${SHA256}  $1" | sha256sum --check --status
}

if verify "$target"; then
    echo "$target is already present and matches the expected checksum."
    exit 0
fi
if [ -e "$target" ]; then
    echo "warning: $target exists but does not match the expected checksum; re-downloading." >&2
fi

echo "Downloading $ASSET (261 MB) from $REPO $TAG ..."
# Download beside the target so an interrupted transfer never leaves a truncated
# file where the tool expects usable weights.
tmp="${target}.partial"
trap 'rm -f "$tmp"' EXIT
curl --fail --location --progress-bar --output "$tmp" "$url"

if ! verify "$tmp"; then
    echo "error: checksum mismatch after download." >&2
    echo "  expected ${SHA256}" >&2
    echo "  got      $(sha256sum "$tmp" | cut -d' ' -f1)" >&2
    exit 1
fi

mv "$tmp" "$target"
trap - EXIT
echo "Wrote $target"
