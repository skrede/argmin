#!/usr/bin/env bash
# Deletability invariant: nothing under tests/ or lib/ may #include a header from
# an optional tree, or CMake-depend on it, so deleting that tree can never break
# the test or library build. The reverse direction -- the publication gate in
# benchmarks/ invoking the re-homed regression_check target under tests/ -- is
# allowed and is not flagged. The check matches real edges (includes, CMake
# target/dir dependencies), not prose comments that merely name a file in one of
# these trees for context.
#
#   scripts/dependency_edge_check.sh [tree ...]
set -euo pipefail

cd "$(dirname "$0")/.."

trees=("$@")
if [ "${#trees[@]}" -eq 0 ]; then
    trees=(benchmarks python)
fi

for tree in "${trees[@]}"; do
    if grep -rnE "#[[:space:]]*include[[:space:]]*[<\"][^>\"]*${tree}/" \
         tests lib --include='*.h' --include='*.hpp' --include='*.cpp'; then
        echo "error: a tests/ or lib/ source includes a ${tree}-tree header"
        exit 1
    fi
    # target_link_libraries(...) et al. are routinely written across
    # multiple physical lines, so match on the logical command (comments
    # stripped, newlines folded) rather than per line -- a line-based grep
    # would pass an idiomatic multi-line re-coupling green.
    while IFS= read -r cml; do
        if sed 's/#.*$//' "$cml" | tr '\n' ' ' \
             | grep -qE "(add_subdirectory|target_link_libraries|target_sources|target_include_directories)[[:space:]]*\([^)]*${tree}"; then
            echo "error: $cml depends on the ${tree} tree"
            exit 1
        fi
    done < <(find tests lib -name CMakeLists.txt)
    echo "no tests|lib -> ${tree} edge"
done
