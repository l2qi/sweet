# macOS Seatbelt (SBPL) reference

Source: `crates/sweet-sandbox/src/seatbelt/mod.rs`. The runner generates a fresh
SBPL profile per command and runs `sandbox-exec -p <profile> -- /bin/bash -c
<command>`. `sandbox-exec` ships with every macOS install — no external
dependency.

## Profile structure (last-match-wins)

SBPL evaluates rules top-to-bottom and the **last matching rule wins**. The
profile therefore starts with a blanket deny and layers specific allows after
it:

```
(version 1)
(deny default)
(allow file-read-metadata)
(allow file-read-data (literal "/"))   ; root inode only — see below
(allow sysctl-read)
... specific allows ...
(deny network*)  |  (allow network*)   ; last, per policy
```

Consequence: **never reorder these or insert a rule above `(deny default)`**,
and beware adding an `allow` that a later rule shadows.

## What each rule is for

| Rule | Why |
|---|---|
| `(deny default)` | baseline; everything denied unless re-allowed below |
| `(allow file-read-metadata)` | `stat()` etc. — broadly needed, leaks no contents |
| `(allow file-read-data (literal "/"))` | dyld reads the **root directory inode itself** (not descendants) during init; without it `bash` dies with SIGABRT |
| `(allow sysctl-read)` | silences lockdown-mode / bootargs probes `bash` does at startup (denials were noisy, non-fatal) |
| `(allow file-read-data (subpath …))` × system paths | read contents of `/usr /bin /sbin /etc /private/etc /tmp /private/tmp /private/var /var /opt /System /Library /Applications /Developer` |
| `(allow file-read-data (subpath …))` × project roots | read the project root(s) |
| `(allow file-read-data (subpath …))` × tool roots | read cargo/rustup/node/etc. (from `tool_paths::resolve_tool_roots`) |
| `(allow file-read-data (literal …))` × safe config files | individual `$HOME` files (`~/.gitconfig`, …) — literal so only the file, not its dir |
| `(allow process-exec (subpath …))` × system + project + tool roots | execute binaries from those trees |
| `(allow process-fork)` | bash subshells, pipes |
| `(allow signal)` | a process must `kill()` its own children; without it `cargo test` leaks fixture subprocesses that hold the stdout pipe open and hang consumers |
| `(allow mach-lookup)` | DNS resolution, system services |
| `(allow file-write* (subpath …))` × project roots | writes confined to the project |
| `(allow file-write* (subpath …))` × `/tmp`, `/private/tmp`, `/var/folders`, `/private/var/folders` | temp dirs compilers/tools need |
| `(allow file-read* file-write* (literal …))` × device nodes | `/dev/null /dev/zero /dev/random /dev/urandom /dev/tty /dev/dtracehelper` |
| `(allow file-read* file-write* (subpath "/dev/fd"))` | `/dev/fd` + `/dev/std{in,out,err}` for process substitution |
| `(deny network*)` / `(allow network*)` | binary, set last, from `SandboxPolicy` |

## Device nodes: never `(subpath "/dev")`

`/dev` contains block devices (`/dev/disk*`, `/dev/rdisk*`) that bypass the
sandbox. Only the safe character devices above are allowlisted individually. Add
new nodes to the explicit list; never open `/dev` wholesale.

## Why `/bin/bash` is hardcoded

PATH lookup resolves to `/opt/homebrew/bin/bash` on Apple Silicon, which is
outside the allowed roots and would be denied by `process-exec`. The runner
invokes the system `/bin/bash` explicitly.

## Escaping

`escape_sbpl` escapes `\`, `"`, `\n`, `\r`, `\t` before embedding any path in a
double-quoted SBPL string. New string interpolation into the profile must go
through it.

## Nesting limitation

`sandbox-exec` cannot nest — launching a profile from inside an existing profile
fails with "Operation not permitted". The test module probes this with
`is_nested_sandbox()` (attempts a no-op `sandbox-exec`) and skips runner tests
when already sandboxed. If you run the suite inside a sandbox those tests will
silently pass-by-skip; verify them in a plain shell.

## Lifecycle

`cmd.kill_on_drop(true)` — if the driving future is dropped (turn cancelled),
`sandbox-exec` is SIGKILLed rather than orphaned to launchd.
