#!/usr/bin/env bash
# Bisect a command that fails only under the sandbox.
#
#   ./repro.sh '<shell command>'
#
# Runs the command three ways and reports the exit code of each:
#   1. unsandboxed              — baseline
#   2. permissive sandbox       — sandbox active, everything allowed
#   3. deny-default sandbox     — sandbox active, only system + CWD reads
#
# Interpreting results:
#   - fails at (1) too        -> it's the command, not the sandbox.
#   - passes (2), fails (3)   -> a missing allow/mount. Widen the real profile
#                                in seatbelt/mod.rs or bubblewrap/mod.rs and add
#                                a regression test.
#   - passes (3) too          -> the real runner adds a rule that breaks it;
#                                compare against generate_profile / build_args.
#
# These profiles APPROXIMATE the real runner. Once you know what's missing, fix
# it in the Rust source (which is the source of truth) and add a test.
set -uo pipefail

[ $# -ge 1 ] || { echo "usage: $0 '<shell command>'" >&2; exit 2; }
CMD="$*"
CWD="$(pwd)"

run() { ( eval "$2" ) >/dev/null 2>&1; echo "  [$1] exit=$?"; }

echo "Command: $CMD"
echo "CWD:     $CWD"
echo

case "$(uname -s)" in
  Darwin)
    run "unsandboxed " "/bin/bash -c \"\$CMD\""

    permissive='(version 1)(allow default)'
    run "permissive  " "sandbox-exec -p '$permissive' -- /bin/bash -c \"\$CMD\""

    # Mirror the real profile's shape: deny default, allow system reads + CWD.
    deny=$(cat <<EOF
(version 1)
(deny default)
(allow file-read-metadata)
(allow file-read-data (literal "/"))
(allow sysctl-read)
(allow process-fork)
(allow signal)
(allow mach-lookup)
(allow network*)
(allow file-read-data (subpath "/usr"))
(allow file-read-data (subpath "/bin"))
(allow file-read-data (subpath "/sbin"))
(allow file-read-data (subpath "/etc"))
(allow file-read-data (subpath "/private"))
(allow file-read-data (subpath "/var"))
(allow file-read-data (subpath "/System"))
(allow file-read-data (subpath "/Library"))
(allow file-read-data (subpath "$CWD"))
(allow process-exec (subpath "/usr"))
(allow process-exec (subpath "/bin"))
(allow process-exec (subpath "/sbin"))
(allow file-write* (subpath "$CWD"))
(allow file-write* (subpath "/tmp"))
(allow file-write* (subpath "/private"))
EOF
)
    run "deny-default" "sandbox-exec -p '$deny' -- /bin/bash -c \"\$CMD\""
    ;;
  Linux)
    if ! command -v bwrap >/dev/null 2>&1; then
      echo "bwrap not installed — install bubblewrap to reproduce the Linux runner." >&2
      run "unsandboxed " "bash -c \"\$CMD\""
      exit 0
    fi
    run "unsandboxed " "bash -c \"\$CMD\""

    run "permissive  " "bwrap --ro-bind / / --bind '$CWD' '$CWD' --tmpfs /tmp --dev /dev --proc /proc --chdir '$CWD' -- bash -c \"\$CMD\""

    # deny-default approximation: ro-bind /, hide $HOME, only re-bind CWD.
    run "deny-default" "bwrap --ro-bind / / --tmpfs \"\$HOME\" --bind '$CWD' '$CWD' --tmpfs /tmp --dev /dev --proc /proc --chdir '$CWD' -- bash -c \"\$CMD\""
    ;;
  *)
    echo "Unsupported platform: $(uname -s)" >&2
    exit 1
    ;;
esac

echo
echo "Source of truth: crates/sweet-sandbox/src/{seatbelt,bubblewrap}/mod.rs"
