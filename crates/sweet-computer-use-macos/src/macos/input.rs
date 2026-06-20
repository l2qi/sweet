// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic mouse, keyboard, and scroll input via Quartz Event Services.
//!
//! Posting events requires the host process to be trusted for Accessibility;
//! the caller checks that before invoking these (events are silently dropped by
//! the OS otherwise).

use std::os::raw::c_ulong;
use std::ptr;

use sweet_computer_use_core::{ComputerUseError, MouseButton, Point};

use super::ffi::{self, CGEventRef, CGPoint, CfBox};

/// Post an event to the HID tap and release it.
fn post(event: CGEventRef) {
    if event.is_null() {
        return;
    }
    // Safety: `event` is a valid +1 CGEvent we own; post then release once.
    unsafe {
        ffi::CGEventPost(ffi::kCGHIDEventTap, event);
        ffi::CFRelease(event);
    }
}

/// Create, optionally tag with a click-state, and post a mouse event.
fn mouse_event(event_type: u32, x: f64, y: f64, button: u32, click_state: Option<i64>) {
    let p = CGPoint { x, y };
    // Safety: null source is valid; struct passed by value per ABI.
    let ev = unsafe { ffi::CGEventCreateMouseEvent(ptr::null(), event_type, p, button) };
    if ev.is_null() {
        return;
    }
    if let Some(state) = click_state {
        // Safety: `ev` is a valid mouse event.
        unsafe { ffi::CGEventSetIntegerValueField(ev, ffi::kCGMouseEventClickState, state) };
    }
    post(ev);
}

/// Current cursor position, read from a synthesized null event.
pub fn cursor_position() -> Option<Point> {
    let ev = CfBox::from_create(unsafe { ffi::CGEventCreate(ptr::null()) })?;
    let p = unsafe { ffi::CGEventGetLocation(ev.as_ptr()) };
    Some(Point { x: p.x, y: p.y })
}

pub fn click(x: f64, y: f64, button: MouseButton) {
    let (down, up, btn) = match button {
        MouseButton::Left => (
            ffi::kCGEventLeftMouseDown,
            ffi::kCGEventLeftMouseUp,
            ffi::kCGMouseButtonLeft,
        ),
        MouseButton::Right => (
            ffi::kCGEventRightMouseDown,
            ffi::kCGEventRightMouseUp,
            ffi::kCGMouseButtonRight,
        ),
        MouseButton::Middle => (
            ffi::kCGEventOtherMouseDown,
            ffi::kCGEventOtherMouseUp,
            ffi::kCGMouseButtonCenter,
        ),
    };
    mouse_event(down, x, y, btn, None);
    mouse_event(up, x, y, btn, None);
}

pub fn double_click(x: f64, y: f64) {
    // A double click is two down/up pairs at the same point; the second pair
    // carries click-state 2 so apps recognize it as a double.
    mouse_event(
        ffi::kCGEventLeftMouseDown,
        x,
        y,
        ffi::kCGMouseButtonLeft,
        Some(1),
    );
    mouse_event(
        ffi::kCGEventLeftMouseUp,
        x,
        y,
        ffi::kCGMouseButtonLeft,
        Some(1),
    );
    mouse_event(
        ffi::kCGEventLeftMouseDown,
        x,
        y,
        ffi::kCGMouseButtonLeft,
        Some(2),
    );
    mouse_event(
        ffi::kCGEventLeftMouseUp,
        x,
        y,
        ffi::kCGMouseButtonLeft,
        Some(2),
    );
}

pub fn move_cursor(x: f64, y: f64) {
    mouse_event(ffi::kCGEventMouseMoved, x, y, ffi::kCGMouseButtonLeft, None);
}

pub fn scroll(x: f64, y: f64, dx: f64, dy: f64) {
    // Scroll applies at the cursor, so place it first.
    unsafe { ffi::CGWarpMouseCursorPosition(CGPoint { x, y }) };
    // wheel1 = vertical, wheel2 = horizontal (line units).
    let ev = unsafe {
        ffi::CGEventCreateScrollWheelEvent(
            ptr::null(),
            ffi::kCGScrollEventUnitLine,
            2,
            dy as i32,
            dx as i32,
        )
    };
    post(ev);
}

pub fn drag(from_x: f64, from_y: f64, to_x: f64, to_y: f64) {
    mouse_event(
        ffi::kCGEventLeftMouseDown,
        from_x,
        from_y,
        ffi::kCGMouseButtonLeft,
        None,
    );
    // Interpolate so apps tracking the drag see intermediate motion.
    const STEPS: usize = 10;
    for i in 1..=STEPS {
        let t = i as f64 / STEPS as f64;
        let x = from_x + (to_x - from_x) * t;
        let y = from_y + (to_y - from_y) * t;
        mouse_event(
            ffi::kCGEventLeftMouseDragged,
            x,
            y,
            ffi::kCGMouseButtonLeft,
            None,
        );
    }
    mouse_event(
        ffi::kCGEventLeftMouseUp,
        to_x,
        to_y,
        ffi::kCGMouseButtonLeft,
        None,
    );
}

/// Type a Unicode string as a single keyboard event pair.
///
/// The string rides on the key-down event only; the key-up is a bare release.
/// Setting the string on both can make some apps insert the text twice.
pub fn type_text(text: &str) {
    let utf16: Vec<u16> = text.encode_utf16().collect();

    let down = unsafe { ffi::CGEventCreateKeyboardEvent(ptr::null(), 0, true) };
    if !down.is_null() {
        // Safety: `down` is valid; `utf16` outlives the call.
        unsafe {
            ffi::CGEventKeyboardSetUnicodeString(down, utf16.len() as c_ulong, utf16.as_ptr());
        }
        post(down);
    }

    let up = unsafe { ffi::CGEventCreateKeyboardEvent(ptr::null(), 0, false) };
    post(up);
}

/// Press a key chord, e.g. `["cmd", "l"]`. Modifier names become event flags
/// applied to each remaining (non-modifier) key's down/up pair.
pub fn key_chord(keys: &[String]) -> Result<(), ComputerUseError> {
    let mut flags: u64 = 0;
    let mut main_keys: Vec<u16> = Vec::new();
    for k in keys {
        let lk = k.to_lowercase();
        if let Some(flag) = modifier_flag(&lk) {
            flags |= flag;
        } else if let Some(code) = keycode(&lk) {
            main_keys.push(code);
        } else {
            return Err(ComputerUseError::InvalidAction(format!(
                "unknown key: {k:?}"
            )));
        }
    }
    if main_keys.is_empty() {
        return Err(ComputerUseError::InvalidAction(
            "key chord has no non-modifier key".to_string(),
        ));
    }
    for code in main_keys {
        for key_down in [true, false] {
            let ev = unsafe { ffi::CGEventCreateKeyboardEvent(ptr::null(), code, key_down) };
            if ev.is_null() {
                continue;
            }
            unsafe { ffi::CGEventSetFlags(ev, flags) };
            post(ev);
        }
    }
    Ok(())
}

/// Map a modifier name to its `CGEventFlags` mask, or `None` if not a modifier.
fn modifier_flag(name: &str) -> Option<u64> {
    Some(match name {
        "cmd" | "command" | "meta" | "super" | "win" => ffi::kCGEventFlagMaskCommand,
        "shift" => ffi::kCGEventFlagMaskShift,
        "alt" | "option" | "opt" => ffi::kCGEventFlagMaskAlternate,
        "ctrl" | "control" => ffi::kCGEventFlagMaskControl,
        "fn" | "function" => ffi::kCGEventFlagMaskSecondaryFn,
        _ => return None,
    })
}

/// Map a key name to its US-ANSI virtual keycode (`kVK_*` from HIToolbox).
fn keycode(name: &str) -> Option<u16> {
    let code = match name {
        "a" => 0,
        "s" => 1,
        "d" => 2,
        "f" => 3,
        "h" => 4,
        "g" => 5,
        "z" => 6,
        "x" => 7,
        "c" => 8,
        "v" => 9,
        "b" => 11,
        "q" => 12,
        "w" => 13,
        "e" => 14,
        "r" => 15,
        "y" => 16,
        "t" => 17,
        "1" => 18,
        "2" => 19,
        "3" => 20,
        "4" => 21,
        "6" => 22,
        "5" => 23,
        "=" | "equal" => 24,
        "9" => 25,
        "7" => 26,
        "-" | "minus" => 27,
        "8" => 28,
        "0" => 29,
        "]" | "rightbracket" => 30,
        "o" => 31,
        "u" => 32,
        "[" | "leftbracket" => 33,
        "i" => 34,
        "p" => 35,
        "return" | "enter" => 36,
        "l" => 37,
        "j" => 38,
        "'" | "quote" => 39,
        "k" => 40,
        ";" | "semicolon" => 41,
        "\\" | "backslash" => 42,
        "," | "comma" => 43,
        "/" | "slash" => 44,
        "n" => 45,
        "m" => 46,
        "." | "period" => 47,
        "tab" => 48,
        "space" | " " => 49,
        "`" | "grave" | "backtick" => 50,
        "delete" | "backspace" => 51,
        "escape" | "esc" => 53,
        "f17" => 64,
        "f18" => 79,
        "f19" => 80,
        "f1" => 122,
        "f2" => 120,
        "f3" => 99,
        "f4" => 118,
        "f5" => 96,
        "f6" => 97,
        "f7" => 98,
        "f8" => 100,
        "f9" => 101,
        "f10" => 109,
        "f11" => 103,
        "f12" => 111,
        "f13" => 105,
        "f14" => 107,
        "f15" => 113,
        "f16" => 106,
        "help" | "insert" => 114,
        "home" => 115,
        "pageup" | "pgup" => 116,
        "forwarddelete" | "del" => 117,
        "end" => 119,
        "pagedown" | "pgdn" => 121,
        "left" => 123,
        "right" => 124,
        "down" => 125,
        "up" => 126,
        _ => return None,
    };
    Some(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifiers_are_recognized() {
        assert_eq!(modifier_flag("cmd"), Some(ffi::kCGEventFlagMaskCommand));
        assert_eq!(modifier_flag("command"), Some(ffi::kCGEventFlagMaskCommand));
        assert_eq!(
            modifier_flag("option"),
            Some(ffi::kCGEventFlagMaskAlternate)
        );
        assert_eq!(modifier_flag("l"), None);
    }

    #[test]
    fn common_keycodes_resolve() {
        assert_eq!(keycode("l"), Some(37));
        assert_eq!(keycode("return"), Some(36));
        assert_eq!(keycode("space"), Some(49));
        assert_eq!(keycode("up"), Some(126));
        assert_eq!(keycode("nope"), None);
    }
}
