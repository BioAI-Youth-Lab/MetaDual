#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for variant in full no_modifiedt_1 no_modifiedt_2 no_structure; do
    python -m model.train --variant "$variant" "$@"
done
