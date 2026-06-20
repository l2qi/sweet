// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end behavioural tests for the platform sandbox.
//!
//! Each test constructs a real [`OsSandbox`] - Seatbelt on macOS, Bubblewrap
//! on Linux - and exercises both the [`CommandRunner`] (sandboxed `bash`)
//! and the [`Filesystem`] (`RestrictedFs`) interfaces, so we can observe what
//! the kernel-level enforcement actually denies. Tests skip gracefully if
//! the platform runner isn't available (e.g. no `bwrap` on a Linux box).
//!
//! Network tests reach `https://example.com` and are skipped if the host has
//! no outbound connectivity.

use std::path::{Path, PathBuf};

use sweet_core::sandbox::{Sandbox, SandboxPolicy};
use sweet_sandbox::OsSandbox;
use tempfile::TempDir;

struct Harness {
    sandbox: OsSandbox,
    project_root: PathBuf,
    outside_root: PathBuf,
    _project: TempDir,
    _outside: TempDir,
}

fn try_harness(policy: SandboxPolicy) -> Option<Harness> {
    let home = std::env::var("HOME").ok()?;
    let home = PathBuf::from(home);

    // Place both trees under $HOME so they aren't in the system read-allow
    // list (`/tmp`, `/var/folders/...`) - that keeps "outside the project
    // root" actually outside any sandbox-permitted region on macOS.
    let project = TempDir::new_in(&home).ok()?;
    let outside = TempDir::new_in(&home).ok()?;

    let project_root = dunce::canonicalize(project.path()).ok()?;
    let outside_root = dunce::canonicalize(outside.path()).ok()?;

    std::fs::write(project_root.join("inside.txt"), b"INSIDE_MARKER\n").ok()?;
    std::fs::write(outside_root.join("secret.txt"), b"OUTSIDE_SECRET\n").ok()?;

    let sandbox = OsSandbox::new(project_root.clone(), policy, Vec::new(), Vec::new()).ok()?;

    Some(Harness {
        sandbox,
        project_root,
        outside_root,
        _project: project,
        _outside: outside,
    })
}

macro_rules! harness_or_skip {
    ($net:expr) => {
        match try_harness($net) {
            Some(h) => h,
            None => {
                eprintln!("skipping: OS sandbox unavailable (missing $HOME or platform runner)");
                return;
            }
        }
    };
}

async fn has_outbound_internet() -> bool {
    let probe = tokio::process::Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            "5",
            "-o",
            "/dev/null",
            "https://example.com",
        ])
        .output()
        .await;
    matches!(probe, Ok(o) if o.status.success())
}

// ---------------------------------------------------------------------------
// Runner: filesystem reads
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_reads_project_file() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("cat inside.txt", Some(&h.project_root), None)
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "INSIDE_MARKER");
}

#[tokio::test]
async fn runner_reads_system_binary_path() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run(
            "/bin/ls /usr/bin > /dev/null && echo ok",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "ok");
}

#[tokio::test]
async fn runner_denies_reads_outside_project() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let secret = h.outside_root.join("secret.txt");
    let cmd = format!("cat {} 2>&1 | head -1; echo END", secret.to_string_lossy());
    let out = h
        .sandbox
        .runner()
        .run(&cmd, Some(&h.project_root), None)
        .await
        .unwrap();
    assert!(
        out.stdout.contains("END"),
        "bash should keep running; stderr: {}, stdout: {}",
        out.stderr,
        out.stdout
    );
    assert!(
        !out.stdout.contains("OUTSIDE_SECRET"),
        "file outside project must not be readable; output: {:?}",
        out.stdout
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn runner_denies_listing_home_on_macos() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("ls -1 ~ 2>&1; echo END", Some(&h.project_root), None)
        .await
        .unwrap();
    assert_ne!(
        out.exit_code, -1,
        "bash was killed by signal; stdout={:?} stderr={:?}",
        out.stdout, out.stderr
    );
    assert!(
        out.stdout.contains("END"),
        "bash didn't survive; stdout={:?} stderr={:?}",
        out.stdout,
        out.stderr
    );
    let lower = out.stdout.to_lowercase();
    assert!(
        lower.contains("operation not permitted") || lower.contains("permission denied"),
        "expected denial in output; got: {}",
        out.stdout
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn runner_home_hides_secret_dirs_on_linux() {
    // bwrap masks $HOME with a tmpfs and re-mounts only the tool paths.
    // Secret dirs like .ssh / .aws / .gnupg must never appear.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("ls -a ~/ 2>&1; echo END", Some(&h.project_root), None)
        .await
        .unwrap();
    assert!(out.stdout.contains("END"));
    for secret in &[".ssh", ".aws", ".gnupg"] {
        assert!(
            !out.stdout.split_whitespace().any(|t| t == *secret),
            "secret dir {secret} leaked into sandboxed $HOME: {}",
            out.stdout
        );
    }
}

// ---------------------------------------------------------------------------
// Runner: filesystem writes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_writes_inside_project() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run(
            "echo WRITTEN > out.txt && cat out.txt",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "WRITTEN");
    let host_view = std::fs::read_to_string(h.project_root.join("out.txt")).unwrap();
    assert_eq!(host_view.trim(), "WRITTEN");
}

#[tokio::test]
async fn runner_denies_writes_outside_project() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let target = h.outside_root.join("evil.txt");
    let cmd = format!(
        "echo NEW_CONTENT > {} 2>&1; echo END",
        target.to_string_lossy()
    );
    let out = h
        .sandbox
        .runner()
        .run(&cmd, Some(&h.project_root), None)
        .await
        .unwrap();
    assert!(out.stdout.contains("END"));
    // From the host's perspective the file must not have been overwritten
    // with the sandboxed write.
    let host_view = std::fs::read_to_string(&target).unwrap_or_default();
    assert!(
        !host_view.contains("NEW_CONTENT"),
        "sandboxed write leaked outside project root: {host_view:?}"
    );
}

// ---------------------------------------------------------------------------
// Runner: process exec
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_can_exec_system_tools() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("/bin/echo hello-from-bin-echo", Some(&h.project_root), None)
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "hello-from-bin-echo");
}

// ---------------------------------------------------------------------------
// Runner: network policy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn network_allow_reaches_internet() {
    if !has_outbound_internet().await {
        eprintln!("skipping: no outbound internet on this host");
        return;
    }
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run(
            "curl -sS --max-time 10 https://example.com | head -1",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.to_lowercase().contains("html"),
        "expected HTML body, got: {:?}",
        out.stdout
    );
}

#[tokio::test]
async fn network_restricted_blocks_by_default() {
    let h = harness_or_skip!(SandboxPolicy::Restricted);
    let out = h
        .sandbox
        .runner()
        .run(
            "curl -sS --max-time 5 -o /dev/null https://example.com; echo exit=$?",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert!(
        out.stdout.contains("exit=") && !out.stdout.contains("exit=0"),
        "curl should fail in restricted mode; stdout: {:?}",
        out.stdout
    );
}

// Note: network policy is fixed at sandbox construction. There is no
// runtime escape hatch in either backend - a user who restricted network at
// startup must restart without the deny flag to re-enable it. The two cases
// above (`network_allow_reaches_internet`, `network_restricted_blocks_by_default`)
// cover the entire policy surface.

// ---------------------------------------------------------------------------
// Filesystem (RestrictedFs) - invoked through Sandbox::fs()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fs_reads_project_file() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let bytes = h
        .sandbox
        .fs()
        .read(&h.project_root.join("inside.txt"))
        .await
        .unwrap();
    assert_eq!(bytes, b"INSIDE_MARKER\n");
}

#[tokio::test]
async fn fs_reads_system_path_metadata() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let meta = h.sandbox.fs().metadata(Path::new("/usr/bin")).await;
    assert!(
        meta.is_ok(),
        "/usr/bin should be readable through RestrictedFs: {:?}",
        meta.err()
    );
}

#[tokio::test]
async fn fs_denies_reads_outside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let result = h
        .sandbox
        .fs()
        .read(&h.outside_root.join("secret.txt"))
        .await;
    assert!(
        result.is_err(),
        "RestrictedFs::read outside project root must fail"
    );
}

#[tokio::test]
async fn fs_writes_inside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let path = h.project_root.join("written_via_fs.txt");
    h.sandbox.fs().write(&path, b"hi").await.unwrap();
    let host_view = std::fs::read_to_string(&path).unwrap();
    assert_eq!(host_view, "hi");
}

#[tokio::test]
async fn fs_denies_writes_outside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let result = h
        .sandbox
        .fs()
        .write(&h.outside_root.join("evil.txt"), b"evil")
        .await;
    assert!(
        result.is_err(),
        "RestrictedFs::write outside root must fail"
    );
    let host_view = std::fs::read_to_string(h.outside_root.join("evil.txt")).unwrap_or_default();
    assert!(
        !host_view.contains("evil"),
        "write should not have reached the host filesystem"
    );
}

#[tokio::test]
async fn fs_denies_dotdot_traversal_out_of_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let outside_name = h.outside_root.file_name().unwrap();
    let traversal = h
        .project_root
        .join("..")
        .join(outside_name)
        .join("secret.txt");
    let result = h.sandbox.fs().read(&traversal).await;
    assert!(
        result.is_err(),
        "dot-dot traversal out of project root must be denied"
    );
}

#[tokio::test]
async fn fs_lists_project_directory() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let entries = h.sandbox.fs().list_dir(&h.project_root).await.unwrap();
    let names: Vec<_> = entries
        .iter()
        .map(|e| {
            e.path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
        .collect();
    assert!(
        names.iter().any(|n| n == "inside.txt"),
        "expected inside.txt in list_dir output: {names:?}"
    );
}

#[tokio::test]
async fn fs_denies_listing_outside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let result = h.sandbox.fs().list_dir(&h.outside_root).await;
    assert!(
        result.is_err(),
        "RestrictedFs::list_dir outside root must fail"
    );
}
