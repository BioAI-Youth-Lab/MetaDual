#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for variant in full abundance_only function_only no_structure no_modifiedt_1 no_modifiedt_2; do
    python -m model.train --variant "$variant" "$@"
done
