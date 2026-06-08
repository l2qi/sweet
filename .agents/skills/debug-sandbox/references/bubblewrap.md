# Linux Bubblewrap (bwrap) reference

Source: `crates/sweet-sandbox/src/bubblewrap/mod.rs`. The runner builds a `bwrap`
argument list per command and runs `bwrap <args> -- bash -c <command>`.

## External dependency + fallback

`bwrap` must be on `$PATH`. `BubblewrapRunner::new` calls `check_bwrap_available`
(`which::which("bwrap")`) at construction and returns
`SandboxError::Backend(<install instructions>)` if absent. `OsSandbox::build`
propagates that error; the consumer decides what to do with it — `shirl-cli`, for
example, catches it and falls back to `DirectSandbox` (unsandboxed) with a
warning. Install:

```
Debian/Ubuntu: apt install bubblewrap
Fedora/RHEL:   dnf install bubblewrap
Arch:          pacman -S bubblewrap
```

## Mount model (order matters)

`build_args` layers the filesystem view bottom-up:

1. `--ro-bind / /` — entire host filesystem, **read-only**, as the baseline.
2. `--tmpfs $HOME` — overlay an empty tmpfs over the home dir, hiding all
   secrets. (Skipped if `$HOME` is unset.)
3. `--ro-bind <tool_root> <tool_root>` for each `resolve_tool_roots()` entry
   that exists — re-expose cargo/rustup/node/etc. **on top of** the tmpfs.
4. `--ro-bind <file> <file>` for each safe config file (`~/.gitconfig`, …) —
   single files, not dirs.
5. `--bind <project_root> <project_root>` — the only **writable** mount.
6. `--tmpfs /tmp` — writable temp (compilers, package managers).
7. `--dev /dev`, `--proc /proc` — minimal device/proc for basic functionality.
8. network: `--unshare-net` if `Restricted`, nothing if `Sandbox`.
9. `--die-with-parent` — don't leak sandboxed processes if the parent dies.
10. `--chdir <cwd>` if an absolute cwd was given.
11. `-- bash -c <command>`.

## The re-exposure hazard

Steps 1–3 are why `tool_paths::resolve_tool_roots` must **never** return `/` or
`$HOME` (or any ancestor of `$HOME`): re-binding such a path in step 3 would
mount the real home directory back on top of the `--tmpfs $HOME` overlay,
defeating the whole isolation. `/bin` and `/sbin` are in nearly every `$PATH`,
and their parent is `/`, so the filter explicitly drops `/`, `$HOME`, ancestors
of `$HOME`, the secret dirs, and any path that contains or sits under a secret
dir. The secret-dir list is the built-in `SECRET_DIRS` plus whatever the consumer
passes as `extra_secret_dirs` (e.g. `.shirl`). Regression tests:
`resolve_excludes_filesystem_root`, `resolve_excludes_home_and_ancestors`,
`resolve_excludes_secret_dirs`, `resolve_excludes_caller_supplied_secret_dirs`.

## Network

Binary, like Seatbelt: `--unshare-net` drops the command into a fresh empty
network namespace (no connectivity at all). There is no per-host filtering and
no mid-session toggle — the namespace is set once per command.

## Lifecycle

`cmd.kill_on_drop(true)` SIGKILLs `bwrap` if the driving future is dropped;
`--die-with-parent` covers the case where the parent exits unexpectedly.
