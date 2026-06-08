# Known gaps (not bugs)

Documented limitations of the sweet-sandbox. Don't "fix" these without an
architectural discussion — they are inherent to the enforcement layers.

## In-process tools bypass the network restriction

`Restricted` policy only affects commands run through the sandbox's
`CommandRunner` (e.g. `bash -c "curl …"`). In-process tools like `HttpFetch` and
`WebSearch` call `reqwest` directly **from the host process**, which is never
sandboxed, so they retain network access even under `Restricted`. Closing this
would require sandboxing the host process itself, not just spawned commands.

## No per-host / per-IP network filtering

Neither macOS SBPL nor Linux bwrap can filter network traffic by domain or IP at
the kernel layer:

- SBPL's `remote` host token must be `*` or `localhost`; `(host …)` is not a
  valid keyword. The test `profile_uses_no_unsupported_network_filters` enforces
  that the generated profile never uses these forms.
- bwrap toggles the entire network namespace (`--unshare-net`) — all or nothing.

So `SandboxPolicy` is binary (`Sandbox` = allow all, `Restricted` = deny all).

## Policy is fixed for the session

`SandboxPolicy` is set once at construction and cannot change mid-session — there
is no escape hatch. Changing it requires restarting the process. This is a
deliberate "honest model" choice given the binary network constraint above.

## Runner tests skip inside an existing sandbox (macOS)

`sandbox-exec` cannot nest, so `SeatbeltRunner` tests detect an enclosing
sandbox via `is_nested_sandbox()` and skip. A green run inside a sandbox does not
mean those tests actually executed — run them in a plain shell to exercise them.

## bwrap absence: construction fails, consumer chooses the fallback

If `bwrap` is not installed, `BubblewrapRunner::new` (and thus `OsSandbox::new`)
returns an error rather than silently degrading. The *consumer* decides whether
to fall back to `DirectSandbox` (unsandboxed) — `shirl-cli`, for instance, prints
a warning and runs without OS-level sandboxing. A user who believes they are
sandboxed on Linux but whose consumer fell back is not. (`RestrictedFs` still
applies to file tools; only the shell-command isolation is lost.)
