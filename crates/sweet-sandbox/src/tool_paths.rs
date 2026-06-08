// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Automatic discovery of tool directories that should be readable inside the
//! sandbox.
//!
//! Resolves `$PATH` entries and known tool directories under the home directory,
//! then excludes known secret paths. The result is a set of canonicalized
//! directories that can be mounted read-only (bwrap) or allowed for reads
//! (Seatbelt, RestrictedFs).

use std::path::PathBuf;

use dirs::home_dir;

#[cfg(test)]
use std::path::Path;

/// Directories under `$HOME` that are known to contain development tools.
/// These are allowed for read access so that compilers, language runtimes,
/// and package managers work inside the sandbox.
const KNOWN_TOOL_DIRS: &[&str] = &[
    ".cargo",
    ".rustup",
    ".local",
    ".nvm",
    ".nix-defexpr",
    ".pyenv",
    ".rbenv",
    ".goenv",
    ".deno",
    ".bun",
    // Git configuration directory (contains config, not credentials)
    ".config/git",
    // Language/package caches
    ".cache/pip",
    ".cache/go-build",
    ".npm",
    "go/pkg/mod",
    ".gradle/caches",
];

/// Directories under `$HOME` that contain secrets or credentials.
/// These are **never** included in tool roots, even if they appear in `$PATH`.
///
/// This is the universal baseline. Applications hide their own credential
/// directories (e.g. where they store API keys) by passing `extra_secret_dirs`
/// to [`resolve_tool_roots`] rather than editing this list.
const SECRET_DIRS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".aws",
    ".config/gh",
    ".config/gcloud",
    ".kube",
    ".npmrc",
    ".pypirc",
    ".netrc",
];

/// Individual files under `$HOME` that contain no secrets but are needed
/// by common CLI tools. These are exposed as single-file read rules (not
/// directory mounts) so the rest of `$HOME` stays hidden.
const SAFE_HOME_CONFIG_FILES: &[&str] = &[".gitconfig", ".gitignore_global"];

/// Resolve all tool directories that should be readable inside the sandbox.
///
/// Sources:
/// 1. Parent directories of every entry in `$PATH` (e.g. `~/.cargo/bin` → `~/.cargo`)
/// 2. Known tool directories under the home directory (e.g. `~/.rustup`, `~/.local`)
///
/// Excludes credential directories — the built-in `SECRET_DIRS` plus any
/// home-relative paths in `extra_secret_dirs` (e.g. `".myapp"`) that the
/// caller wants kept out of the sandbox — and deduplicates.
pub fn resolve_tool_roots(extra_secret_dirs: &[String]) -> Vec<PathBuf> {
    let Some(home) = home_dir() else {
        return Vec::new();
    };

    let mut roots: Vec<PathBuf> = Vec::new();

    // 1. $PATH entries — add the parent directory of each entry
    if let Ok(path_var) = std::env::var("PATH") {
        for entry in std::env::split_paths(&path_var) {
            if let Some(parent) = entry.parent() {
                let canonical =
                    dunce::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
                if !roots.contains(&canonical) {
                    roots.push(canonical);
                }
            }
        }
    }

    // 2. Known tool directories under home
    for dir_name in KNOWN_TOOL_DIRS {
        let candidate = home.join(dir_name);
        if candidate.exists() {
            if let Ok(canonical) = dunce::canonicalize(&candidate) {
                if !roots.contains(&canonical) {
                    roots.push(canonical);
                }
            }
        }
    }

    // 3. Filter out unsafe roots.
    //
    // A tool root is unsafe if re-exposing it would defeat the sandbox's
    // home-directory isolation. That means rejecting:
    //
    //   (a) `/` and any other ancestor of home — re-mounting these on
    //       Linux clobbers the `--tmpfs $HOME` overlay, and on macOS grants
    //       blanket read access. `/bin` and `/sbin` are in nearly every
    //       PATH, so their parent (`/`) lands here without this guard.
    //   (b) Home itself — same reasoning.
    //   (c) The secret dirs themselves and anything under them.
    //   (d) Paths that *contain* a secret dir — re-mounting them would
    //       transitively re-expose `~/.ssh`, `~/.aws`, etc.
    let secret_canonical: Vec<PathBuf> = SECRET_DIRS
        .iter()
        .map(|s| s.to_string())
        .chain(extra_secret_dirs.iter().cloned())
        .filter_map(|s| {
            let p = home.join(&s);
            // Canonicalize if it exists, otherwise use the raw path
            dunce::canonicalize(&p).ok().or(Some(p))
        })
        .collect();

    roots.retain(|root| {
        // (a) + (b): reject home itself and any ancestor of it.
        if home.starts_with(root) {
            return false;
        }
        for secret in &secret_canonical {
            // (c): root is the secret itself or sits under it.
            if root == secret || root.starts_with(secret) {
                return false;
            }
            // (d): root contains the secret as a descendant.
            if secret.starts_with(root) {
                return false;
            }
        }
        true
    });

    roots
}

/// Resolve individual config files from the home directory that are safe to expose.
///
/// Unlike tool roots (directories), these are specific files — only the
/// file itself is readable, not its parent directory. Used by Seatbelt
/// `(literal ...)` rules and bwrap `--ro-bind` for individual files.
pub fn resolve_safe_config_files() -> Vec<PathBuf> {
    let Some(home) = home_dir() else {
        return Vec::new();
    };

    SAFE_HOME_CONFIG_FILES
        .iter()
        .filter_map(|name| {
            let path = home.join(name);
            if path.is_file() {
                dunce::canonicalize(&path).ok()
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_excludes_secret_dirs() {
        let home = home_dir().expect("home directory not found");
        // Ensure secret dirs exist for the test
        std::fs::create_dir_all(home.join(".ssh")).unwrap();

        let roots = resolve_tool_roots(&[]);
        let ssh_path = dunce::canonicalize(home.join(".ssh")).unwrap();
        assert!(
            !roots
                .iter()
                .any(|r| r == &ssh_path || r.starts_with(&ssh_path)),
            ".ssh should be excluded from tool roots"
        );
    }

    #[test]
    fn resolve_includes_cargo_if_present() {
        let home = home_dir().expect("home directory not found");
        let cargo_dir = home.join(".cargo");
        if !cargo_dir.exists() {
            return; // skip if not installed
        }

        let roots = resolve_tool_roots(&[]);
        let canonical = dunce::canonicalize(&cargo_dir).unwrap();
        assert!(
            roots.iter().any(|r| r == &canonical),
            ".cargo should be included in tool roots when it exists"
        );
    }

    #[test]
    fn resolve_excludes_caller_supplied_secret_dirs() {
        let home = home_dir().expect("home directory not found");
        let cargo_dir = home.join(".cargo");
        if !cargo_dir.exists() {
            return; // skip if not installed
        }
        let canonical = dunce::canonicalize(&cargo_dir).unwrap();

        // `.cargo` is a known tool dir, so it is a root by default...
        assert!(resolve_tool_roots(&[]).iter().any(|r| r == &canonical));

        // ...but naming it in `extra_secret_dirs` excludes it, proving the
        // caller-supplied denylist is honored alongside the built-ins.
        let roots = resolve_tool_roots(&[".cargo".to_string()]);
        assert!(
            !roots
                .iter()
                .any(|r| r == &canonical || r.starts_with(&canonical)),
            "caller-supplied secret dir should be excluded from tool roots"
        );
    }

    #[test]
    fn resolve_includes_path_parents() {
        let roots = resolve_tool_roots(&[]);
        // At minimum /usr/bin should be in PATH, so /usr should be a root
        assert!(
            roots.iter().any(|r| r == Path::new("/usr")),
            "/usr (parent of /usr/bin from $PATH) should be a tool root"
        );
    }

    /// Regression: `/bin` and `/sbin` are in nearly every PATH, and their
    /// parent is `/`. If `/` leaks into tool_roots, the bubblewrap runner
    /// re-binds `/` on top of `--tmpfs $HOME` and the entire home directory
    /// becomes visible again. RestrictedFs hits the same bug — `/` in the
    /// read-roots list means every path passes the read check.
    #[test]
    fn resolve_excludes_filesystem_root() {
        let roots = resolve_tool_roots(&[]);
        assert!(
            !roots.iter().any(|r| r == Path::new("/")),
            "filesystem root `/` must never appear in tool_roots: {roots:?}"
        );
    }

    /// Regression: the home directory (and any ancestor of it) must not appear
    /// as a tool root, because re-mounting it would re-expose every secret dir
    /// the filter is trying to hide.
    #[test]
    fn resolve_excludes_home_and_ancestors() {
        let home = home_dir().expect("home directory not found");
        let roots = resolve_tool_roots(&[]);
        for root in &roots {
            assert!(
                !home.starts_with(root),
                "tool root {root:?} is an ancestor of home ({home:?})"
            );
        }
    }
}
