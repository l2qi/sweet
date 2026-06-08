#!/usr/bin/env bash
# Show the directories the sandbox would make readable, mirroring
# `tool_paths::resolve_tool_roots(extra_secret_dirs)` in
# crates/sweet-sandbox/src/tool_paths.rs.
#
#   ./show-tool-roots.sh [extra_secret_dir ...]
#
# Pass any home-relative dirs the consumer hides via `extra_secret_dirs` as
# arguments, e.g. `show-tool-roots.sh .shirl` to mirror shirl's configuration.
#
# Use this to answer "will my tool / file be reachable inside the sandbox?"
# A path is reachable iff it lives under one of the roots printed here, under a
# SYSTEM_READ_PATH, or under the project root.
#
# This is a faithful re-implementation of the Rust logic, but the Rust source is
# the source of truth — if they disagree, the Rust wins (and this script is out
# of date).
set -euo pipefail

canon() { realpath "$1" 2>/dev/null || readlink -f "$1" 2>/dev/null || printf '%s' "$1"; }

# Component-aware prefix test: does $1 sit under (or equal) $2?
under() {
  local child="$1" parent="$2"
  [ "$parent" = "/" ] && return 0   # every absolute path is under root
  case "$child" in "$parent") return 0;; "$parent"/*) return 0;; *) return 1;; esac
}

KNOWN_TOOL_DIRS=(
  .cargo .rustup .local .nvm .nix-defexpr .pyenv .rbenv .goenv .deno .bun
  .config/git .cache/pip .cache/go-build .npm go/pkg/mod .gradle/caches
)
# Built-in SECRET_DIRS (universal credential dirs) + any extra dirs the consumer
# supplied as arguments (mirrors the `extra_secret_dirs` parameter).
SECRET_DIRS=(
  .ssh .gnupg .aws .config/gh .config/gcloud .kube .npmrc .pypirc .netrc
  "$@"
)

[ -n "${HOME:-}" ] || { echo "HOME not set — resolve_tool_roots would return empty." >&2; exit 1; }

roots=()
add_root() { local r; r="$(canon "$1")"; for e in "${roots[@]:-}"; do [ "$e" = "$r" ] && return; done; roots+=("$r"); }

# 1. Parent directory of every $PATH entry.
IFS=':' read -ra path_entries <<< "${PATH:-}"
for entry in "${path_entries[@]}"; do
  [ -n "$entry" ] || continue
  add_root "$(dirname "$entry")"
done

# 2. Known tool dirs under $HOME that exist.
for d in "${KNOWN_TOOL_DIRS[@]}"; do
  [ -e "$HOME/$d" ] && add_root "$HOME/$d"
done

# 3. Filter out unsafe roots.
secret_canon=()
for s in "${SECRET_DIRS[@]}"; do secret_canon+=("$(canon "$HOME/$s")"); done

home_canon="$(canon "$HOME")"
kept=()
for root in "${roots[@]}"; do
  # (a)+(b): reject $HOME and any ancestor of it.
  if under "$home_canon" "$root"; then continue; fi
  drop=0
  for secret in "${secret_canon[@]}"; do
    # (c) root is the secret or under it; (d) root contains the secret.
    if [ "$root" = "$secret" ] || under "$root" "$secret" || under "$secret" "$root"; then drop=1; break; fi
  done
  [ "$drop" -eq 1 ] && continue
  kept+=("$root")
done

echo "Tool roots that would be readable inside the sandbox:"
printf '  %s\n' "${kept[@]}"
echo
echo "Plus the SYSTEM_READ_PATHS baseline and the project root (not shown here)."
echo "Secret dirs are excluded by design: ${SECRET_DIRS[*]}"
