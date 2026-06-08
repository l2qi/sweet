---
name: debug-sandbox
description: >-
  Diagnose why a shell command or file operation behaves differently under the
  OS sandbox or network restriction. Use when a command works unsandboxed but
  fails sandboxed, exits with code -1, aborts with SIGABRT, "command not found"
  only when sandboxed, a write/read is denied, curl/network hangs or is blocked,
  cargo test hangs after passing, or bwrap is reported missing. Covers macOS
  Seatbelt (sandbox-exec), Linux Bubblewrap (bwrap), and the in-process
  RestrictedFs layer in crates/sweet-sandbox.
---

# Debugging the sweet-sandbox

Maintainer skill for triaging sandbox behaviour in this repo. The sandbox lives
in `crates/sweet-sandbox/` (`SeatbeltRunner`, `BubblewrapRunner`, `RestrictedFs`,
`OsSandbox`, `tool_paths`) plus the policy/trait definitions in
`crates/sweet-core/src/sandbox.rs`. Read this before changing a profile or a
mount list — the failure modes are non-obvious and several rules exist as
regression guards.

`sweet-sandbox` is a library; it has no CLI. Network/sandbox policy is chosen by
the *consumer* (e.g. the `shirl` coding assistant) and passed in at construction.
Where this skill refers to "the consumer", that is the binary wiring the sandbox.

## First: which layer denied it?

There are **two independent enforcement layers**. A denial comes from exactly
one of them — identify which before touching anything.

1. **`RestrictedFs`** (`crates/sweet-sandbox/src/restricted_fs.rs`) — in-process,
   wraps the `Filesystem` trait. Validates every read/write/remove/rename against
   canonicalized read-roots and write-roots *before* any syscall. A denial here
   surfaces as `SandboxError::PathDenied { path, reason }`. This affects the
   file *tools* (`read_file`, `write_file`, `edit_file`, …), not `bash`.
2. **`CommandRunner`** (`SeatbeltRunner` on macOS / `BubblewrapRunner` on Linux)
   — sandboxes the *shell*. Wraps the command in `sandbox-exec -p <profile> --
   /bin/bash -c …` (macOS) or `bwrap <args> -- bash -c …` (Linux). A denial here
   surfaces as a non-zero (or -1) exit code from the command itself.

If the failing operation is a file tool → look at `RestrictedFs` + `tool_paths`.
If it is a `bash`/shell command → look at the runner's profile/mounts.

`OsSandbox::new(project_root, policy, extra_read_roots, extra_secret_dirs)`
bundles the platform runner + `RestrictedFs` together (see `os_sandbox.rs`); both
share the same read-root logic via `tool_paths::resolve_tool_roots(extra_secret_dirs)`.

- `extra_read_roots` — directories the agent may *read* but not write (e.g. a
  session-state dir outside the project root).
- `extra_secret_dirs` — home-relative dirs to *hide* on top of the built-in
  credential list (`SECRET_DIRS` in `tool_paths.rs`). The framework ships only
  universal secrets (`.ssh`, `.aws`, …); a consumer hides its own credential dir
  by passing e.g. `".myapp"` (shirl passes `".shirl"`, where `auth.toml` lives).

## Policy model — read this before "adding a network rule"

`SandboxPolicy` is `Off | Sandbox | Restricted`, fixed at construction
(`crates/sweet-core/src/sandbox.rs`). The consumer chooses it — typically a
coding assistant maps `--sandbox` → `Sandbox`, `--restrict-network` →
`Restricted` (implies the sandbox), default → `Off`.

**Network filtering is binary — allow all or deny all.** Neither macOS SBPL nor
bwrap can filter by host/IP at the kernel layer. Do **not** attempt `(host …)`,
`(remote ip …)`, `(remote tcp …)`, or per-domain bwrap rules — SBPL rejects them
and there is a test (`profile_uses_no_unsupported_network_filters`) guarding it.
There is no mid-session escape hatch; to change policy the process restarts.

## Symptom → cause → fix

| Symptom | Layer | Cause | Fix / check |
|---|---|---|---|
| `bash` aborts with **SIGABRT** before running anything (macOS) | runner | dyld reads the root inode during init; needs `(allow file-read-data (literal "/"))` | already in `generate_profile`; suspect a rule added *before* `(deny default)` breaking last-match-wins |
| Exit code **-1** | runner | process killed by a signal — `status.code()` returned `None`, not a clean exit | not a permission denial per se; check `(allow signal)` / `(allow process-fork)` are present |
| **"command not found"** / `process-exec` denied (macOS, Apple Silicon) | runner | PATH resolves the tool (e.g. `bash`) to `/opt/homebrew/bin`, outside the allowed roots | runner hardcodes `/bin/bash`; for *other* tools, confirm their dir is a resolved `tool_root` (`scripts/show-tool-roots.sh`) |
| File read works unsandboxed, **denied** sandboxed | either | path is not under a read-root (system paths, tool roots, project root, or a safe config file) | by design — `$HOME` secrets are walled off. If it *should* be reachable, see `tool_paths`; if it's a secret the consumer hid, check `extra_secret_dirs` |
| Write **rejected** (`PathDenied`) | `RestrictedFs` | path is outside the project root (the only write-root) | expected; only the project root + `/tmp` (via runner) are writable |
| `cargo test` **hangs** after tests pass (macOS) | runner | a test binary couldn't `kill()` a leaked fixture (mock server) holding the stdout pipe open | regression guard is `(allow signal)`; see `sandboxed_process_can_kill_its_own_child` |
| Network **blocked** unexpectedly | policy | the consumer constructed `SandboxPolicy::Restricted` | inspect how the consumer built `SandboxPolicy` |
| in-process tools (e.g. `HttpFetch`/`WebSearch`) **still reach** the net under `Restricted` | n/a | in-process `reqwest` from the host bypasses the runner | documented gap, **not a bug** — only `CommandRunner` commands are filtered |
| Runner tests **silently skip** (macOS) | tests | already inside a Seatbelt profile; `sandbox-exec` cannot nest | `is_nested_sandbox()` probe — run them outside any sandbox |
| `OsSandbox::new` **errors** with bwrap install instructions (Linux) | runner | `bwrap` is absent; `BubblewrapRunner::new` returns `SandboxError::Backend` | `apt/dnf/pacman install bubblewrap`, or have the consumer fall back to `DirectSandbox` (shirl-cli does, with a warning) |
| Home directory **visible again** inside the sandbox (Linux) | tool_paths | `/` or `$HOME` leaked into `tool_roots` and got re-bound over the `--tmpfs $HOME` overlay | guarded by `resolve_excludes_filesystem_root` / `resolve_excludes_home_and_ancestors` — don't weaken the filter |

## Three things that are easy to get wrong

1. **SBPL is last-match-wins.** `generate_profile` places `(deny default)`
   first; every `(allow …)` after it wins. Reordering rules, or adding an
   `allow` that a later `deny` shadows, silently breaks the profile. Never
   reorder casually.
2. **`tool_roots` re-exposure.** On Linux the baseline is `--ro-bind / /` then
   `--tmpfs $HOME`, then tool roots are re-bound *on top*. If `/` or `$HOME` (or
   an ancestor) sneaks into `resolve_tool_roots`, the tmpfs overlay is clobbered
   and every secret dir is visible again. The same list feeds `RestrictedFs`
   read-roots, so `/` there means *every* path passes the read check.
3. **Device nodes are allowlisted individually**, not via `(subpath "/dev")` —
   `/dev` contains block devices (`/dev/disk*`) that bypass the sandbox. Add new
   nodes to the explicit list, never open `/dev` wholesale.

## Reproduction workflow

When a command fails only under the sandbox, bisect with `scripts/repro.sh`
(see its header). It runs the command three ways — unsandboxed, under a
permissive profile, and under a deny-default-plus-system-reads profile — so you
can localize the cause:

- fails **unsandboxed too** → it's the command, not the sandbox.
- passes permissive, fails deny-default → a **missing `allow`/mount**; widen the
  profile minimally and add a regression test.
- check `scripts/show-tool-roots.sh` to see exactly which directories the real
  runner would make readable (mirrors `resolve_tool_roots`). Pass any
  `extra_secret_dirs` the consumer uses as arguments, e.g. `show-tool-roots.sh .shirl`.

The scripts are approximations of the generated profile/mounts — once you know
*what* is missing, fix it in the Rust source (`seatbelt/mod.rs`,
`bubblewrap/mod.rs`, or `tool_paths.rs`) and add a test under
`crates/sweet-sandbox/`.

## Deeper references

- `references/seatbelt-sbpl.md` — the macOS profile: every operation it grants
  and why, the device-node list, escaping, the nesting limitation.
- `references/bubblewrap.md` — the Linux mount model and namespace flags.
- `references/known-gaps.md` — documented limitations that are not bugs.

## When you change the sandbox

Per the repo's "What to update when you change things" table (CLAUDE.md): any
sandbox behaviour change needs a hermetic test in `crates/sweet-sandbox/` (see
`tests/sandbox_behaviour.rs` and the per-module `#[cfg(test)]` blocks), and a
public-API change (`OsSandbox::new`, `RestrictedFs`, `resolve_tool_roots`, …)
means updating downstream consumers. Run the full pre-commit checklist
(`/check`). Note runner tests skip inside an existing sandbox — verify them in an
unsandboxed shell.
