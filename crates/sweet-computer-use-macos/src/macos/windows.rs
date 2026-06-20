// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! The on-screen window list via `CGWindowListCopyWindowInfo`.
//!
//! Owner names and bounds are available without Screen Recording permission;
//! window *titles* (`kCGWindowName`) are not, so they may be absent.

use sweet_computer_use_core::{Rect, WindowInfo};

use super::ffi::{self, CGRect, CfBox, Pid};

/// List on-screen windows (desktop elements excluded), front to back.
///
/// `active_pid` marks which app is frontmost: if given, windows owned by that
/// pid are flagged active; otherwise the first normal-layer window (the
/// front-most) is.
pub fn list_windows(active_pid: Option<Pid>) -> Vec<WindowInfo> {
    let option = ffi::kCGWindowListOptionOnScreenOnly | ffi::kCGWindowListExcludeDesktopElements;
    let Some(arr) = CfBox::from_create(unsafe {
        ffi::CGWindowListCopyWindowInfo(option, ffi::kCGNullWindowID)
    }) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut marked_active = false;
    let n = unsafe { ffi::CFArrayGetCount(arr.as_ptr()) };
    for i in 0..n {
        let dict = unsafe { ffi::CFArrayGetValueAtIndex(arr.as_ptr(), i) };
        if dict.is_null() {
            continue;
        }
        let app = dict_string(dict, "kCGWindowOwnerName").unwrap_or_default();
        let title = dict_string(dict, "kCGWindowName").filter(|s| !s.is_empty());
        let bounds = dict_rect(dict, "kCGWindowBounds").unwrap_or_default();
        let window_id = dict_i64(dict, "kCGWindowNumber").unwrap_or(0) as u32;
        let pid = dict_i64(dict, "kCGWindowOwnerPID");
        let layer = dict_i64(dict, "kCGWindowLayer").unwrap_or(0);

        let is_active = match active_pid {
            Some(ap) => pid == Some(ap as i64),
            None => {
                if layer == 0 && !marked_active {
                    marked_active = true;
                    true
                } else {
                    false
                }
            }
        };

        out.push(WindowInfo {
            app,
            title,
            bounds,
            is_active,
            window_id,
        });
    }
    out
}

/// Borrowed lookup into a CG window dictionary (the value is owned by the dict).
fn dict_get(dict: ffi::CFDictionaryRef, key: &str) -> ffi::CFTypeRef {
    let k = ffi::cfstr(key);
    unsafe { ffi::CFDictionaryGetValue(dict, k.as_ptr()) }
}

fn dict_string(dict: ffi::CFDictionaryRef, key: &str) -> Option<String> {
    let v = dict_get(dict, key);
    if ffi::is_string(v) {
        ffi::cf_string_to_string(v)
    } else {
        None
    }
}

fn dict_i64(dict: ffi::CFDictionaryRef, key: &str) -> Option<i64> {
    ffi::cf_i64(dict_get(dict, key))
}

fn dict_rect(dict: ffi::CFDictionaryRef, key: &str) -> Option<Rect> {
    let v = dict_get(dict, key);
    if v.is_null() {
        return None;
    }
    let mut cg = CGRect::default();
    // The bounds value is itself a CFDictionary; CGRectMake... parses it.
    let ok = unsafe { ffi::CGRectMakeWithDictionaryRepresentation(v, &mut cg) };
    if ok {
        Some(Rect {
            x: cg.origin.x,
            y: cg.origin.y,
            width: cg.size.width,
            height: cg.size.height,
        })
    } else {
        None
    }
}
