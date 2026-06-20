// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Accessibility-tree extraction and element actions via `AXUIElement`.

use sweet_computer_use_core::{ComputerUseError, ObserveOptions, Rect, UiNode};

use super::ffi::{self, copy_attr, AXUIElementRef, CfBox, Pid};

/// The frontmost application and its focused window, resolved from the
/// system-wide accessibility element.
pub struct FocusedContext {
    // Held only to keep the references alive; not read directly.
    #[allow(dead_code)]
    system: CfBox,
    #[allow(dead_code)]
    app: CfBox,
    pub window: Option<CfBox>,
    pub app_name: Option<String>,
    pub pid: Option<Pid>,
}

/// Resolve the frontmost app and its focused (or main) window. Returns `None`
/// when nothing is focused or accessibility yields no application.
pub fn focused_context() -> Option<FocusedContext> {
    let system = CfBox::from_create(unsafe { ffi::AXUIElementCreateSystemWide() })?;
    let app = copy_attr(system.as_ptr(), "AXFocusedApplication")?;
    if !ffi::is_ax_element(app.as_ptr()) {
        return None;
    }

    let app_name = attr_string(app.as_ptr(), "AXTitle").filter(|s| !s.is_empty());

    let mut raw_pid: Pid = 0;
    let pid =
        if unsafe { ffi::AXUIElementGetPid(app.as_ptr(), &mut raw_pid) } == ffi::kAXErrorSuccess {
            Some(raw_pid)
        } else {
            None
        };

    let window = copy_attr(app.as_ptr(), "AXFocusedWindow")
        .filter(|w| ffi::is_ax_element(w.as_ptr()))
        .or_else(|| {
            copy_attr(app.as_ptr(), "AXMainWindow").filter(|w| ffi::is_ax_element(w.as_ptr()))
        });

    Some(FocusedContext {
        system,
        app,
        window,
        app_name,
        pid,
    })
}

/// Walk an element into a [`UiNode`], honoring the depth and node-count budget.
///
/// `count` is the running total of emitted nodes (including this one); the walk
/// stops adding children once it reaches `opts.max_nodes`.
pub fn walk(
    el: AXUIElementRef,
    path: String,
    depth: usize,
    opts: &ObserveOptions,
    count: &mut usize,
) -> UiNode {
    let role = attr_string(el, "AXRole").unwrap_or_else(|| "AXUnknown".to_string());
    let title = attr_string(el, "AXTitle").filter(|s| !s.is_empty());
    let label = attr_string(el, "AXDescription").filter(|s| !s.is_empty());
    let value = value_string(el).filter(|s| !s.is_empty());
    let identifier = attr_string(el, "AXIdentifier").filter(|s| !s.is_empty());
    let enabled = copy_attr(el, "AXEnabled")
        .and_then(|b| ffi::cf_bool(b.as_ptr()))
        .unwrap_or(true);
    let focused = copy_attr(el, "AXFocused")
        .and_then(|b| ffi::cf_bool(b.as_ptr()))
        .unwrap_or(false);
    let frame = element_frame(el);
    let actions = action_names(el);

    *count += 1;

    let mut children = Vec::new();
    if depth < opts.max_depth {
        if let Some(arr) = copy_attr(el, "AXChildren") {
            if ffi::is_array(arr.as_ptr()) {
                let n = unsafe { ffi::CFArrayGetCount(arr.as_ptr()) };
                for i in 0..n {
                    if *count >= opts.max_nodes {
                        break;
                    }
                    // Borrowed from `arr`; valid while `arr` is alive (it is,
                    // through the end of this loop).
                    let child = unsafe { ffi::CFArrayGetValueAtIndex(arr.as_ptr(), i) };
                    if !ffi::is_ax_element(child) {
                        continue;
                    }
                    children.push(walk(child, format!("{path}/{i}"), depth + 1, opts, count));
                }
            }
        }
    }

    UiNode {
        path,
        role,
        title,
        label,
        value,
        identifier,
        frame,
        enabled,
        focused,
        actions,
        children,
    }
}

/// Perform `AXPress` on the element at `path` within the focused window.
pub fn press(path: &str) -> Result<(), ComputerUseError> {
    let ctx = focused_context()
        .ok_or_else(|| ComputerUseError::ElementNotFound("no focused window".to_string()))?;
    let window = ctx
        .window
        .as_ref()
        .ok_or_else(|| ComputerUseError::ElementNotFound("no focused window".to_string()))?;
    let action = ffi::cfstr("AXPress");
    let res = with_element(window.as_ptr(), path, |el| unsafe {
        ffi::AXUIElementPerformAction(el, action.as_ptr())
    });
    match res {
        None => Err(ComputerUseError::ElementNotFound(path.to_string())),
        Some(err) if err == ffi::kAXErrorSuccess => Ok(()),
        Some(err) => Err(ComputerUseError::Platform(format!(
            "AXPress failed on {path} (AXError {err})"
        ))),
    }
}

/// Set `AXValue` on the element at `path` within the focused window.
pub fn set_value(path: &str, value: &str) -> Result<(), ComputerUseError> {
    let ctx = focused_context()
        .ok_or_else(|| ComputerUseError::ElementNotFound("no focused window".to_string()))?;
    let window = ctx
        .window
        .as_ref()
        .ok_or_else(|| ComputerUseError::ElementNotFound("no focused window".to_string()))?;
    let attr = ffi::cfstr("AXValue");
    let val = ffi::cfstr(value);
    let res = with_element(window.as_ptr(), path, |el| unsafe {
        ffi::AXUIElementSetAttributeValue(el, attr.as_ptr(), val.as_ptr())
    });
    match res {
        None => Err(ComputerUseError::ElementNotFound(path.to_string())),
        Some(err) if err == ffi::kAXErrorSuccess => Ok(()),
        Some(err) => Err(ComputerUseError::Platform(format!(
            "AXSetValue failed on {path} (AXError {err})"
        ))),
    }
}

/// Resolve `path` (slash-joined child indices, rooted at the window with the
/// leading `0`) to a live element and run `f` against it.
///
/// Returns `None` if the path is malformed or any index is out of range. The
/// recursion keeps each parent's `AXChildren` array alive on its stack frame
/// while descending, so the borrowed child handle stays valid.
pub fn with_element<R>(
    root_window: AXUIElementRef,
    path: &str,
    f: impl FnOnce(AXUIElementRef) -> R,
) -> Option<R> {
    let mut parts = path.split('/');
    if parts.next() != Some("0") {
        return None;
    }
    let indices: Vec<ffi::CFIndex> = parts
        .map(|p| p.parse::<ffi::CFIndex>())
        .collect::<Result<_, _>>()
        .ok()?;
    descend(root_window, &indices, f)
}

fn descend<R>(
    el: AXUIElementRef,
    indices: &[ffi::CFIndex],
    f: impl FnOnce(AXUIElementRef) -> R,
) -> Option<R> {
    match indices.split_first() {
        None => Some(f(el)),
        Some((&idx, rest)) => {
            let arr = copy_attr(el, "AXChildren")?;
            if !ffi::is_array(arr.as_ptr()) {
                return None;
            }
            let count = unsafe { ffi::CFArrayGetCount(arr.as_ptr()) };
            if idx < 0 || idx >= count {
                return None;
            }
            let child = unsafe { ffi::CFArrayGetValueAtIndex(arr.as_ptr(), idx) };
            if !ffi::is_ax_element(child) {
                return None;
            }
            // `arr` stays alive here until `descend` returns, keeping `child` valid.
            descend(child, rest, f)
        }
    }
}

// ---- attribute helpers ----

/// Copy an attribute and, if it is a `CFString`, return its text.
pub fn attr_string(el: AXUIElementRef, attr: &str) -> Option<String> {
    let v = copy_attr(el, attr)?;
    if ffi::is_string(v.as_ptr()) {
        ffi::cf_string_to_string(v.as_ptr())
    } else {
        None
    }
}

/// `AXValue` rendered as text - handles string, boolean, and numeric values.
fn value_string(el: AXUIElementRef) -> Option<String> {
    let v = copy_attr(el, "AXValue")?;
    let p = v.as_ptr();
    if ffi::is_string(p) {
        ffi::cf_string_to_string(p)
    } else if let Some(b) = ffi::cf_bool(p) {
        Some(if b { "true" } else { "false" }.to_string())
    } else {
        ffi::cf_i64(p).map(|n| n.to_string())
    }
}

fn element_frame(el: AXUIElementRef) -> Option<Rect> {
    let pos = copy_attr(el, "AXPosition").and_then(|v| ffi::ax_point(v.as_ptr()))?;
    let size = copy_attr(el, "AXSize").and_then(|v| ffi::ax_size(v.as_ptr()))?;
    Some(Rect {
        x: pos.x,
        y: pos.y,
        width: size.width,
        height: size.height,
    })
}

fn action_names(el: AXUIElementRef) -> Vec<String> {
    let mut names: ffi::CFArrayRef = std::ptr::null();
    let err = unsafe { ffi::AXUIElementCopyActionNames(el, &mut names) };
    if err != ffi::kAXErrorSuccess {
        return Vec::new();
    }
    let Some(arr) = CfBox::from_create(names) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let n = unsafe { ffi::CFArrayGetCount(arr.as_ptr()) };
    for i in 0..n {
        let s = unsafe { ffi::CFArrayGetValueAtIndex(arr.as_ptr(), i) };
        // Guard the type before treating it as a CFString - the array is
        // documented to hold strings, but a type check keeps a malformed
        // element from being type-confused.
        if ffi::is_string(s) {
            if let Some(name) = ffi::cf_string_to_string(s) {
                out.push(name);
            }
        }
    }
    out
}
