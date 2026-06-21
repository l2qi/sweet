// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! macOS-specific implementation, assembled from the submodules.

mod accessibility;
mod ffi;
mod input;
mod permissions;
mod screenshot;
mod windows;

use std::path::Path;

use sweet_computer_use_core::{
    ActionOutcome, ComputerAction, ComputerObservation, ComputerUseError, ObserveOptions, Point,
    Size,
};

/// Build a [`ComputerObservation`] of the current desktop.
pub fn observe(
    screenshot_dir: &Path,
    opts: &ObserveOptions,
) -> Result<ComputerObservation, ComputerUseError> {
    let mut notes = Vec::new();

    let display = unsafe { ffi::CGMainDisplayID() };
    let bounds = unsafe { ffi::CGDisplayBounds(display) };
    let screen_size = Size {
        width: bounds.size.width,
        height: bounds.size.height,
    };

    let cursor = input::cursor_position();

    // Accessibility tree (requires the Accessibility grant).
    let (mut active_app, mut active_window_title, tree, pid) =
        if permissions::accessibility_trusted() {
            match accessibility::focused_context() {
                Some(ctx) => {
                    let title = ctx
                        .window
                        .as_ref()
                        .and_then(|w| accessibility::attr_string(w.as_ptr(), "AXTitle"))
                        .filter(|s| !s.is_empty());
                    let tree = match (&ctx.window, opts.include_tree) {
                        (Some(w), true) => {
                            let mut count = 0usize;
                            Some(accessibility::walk(
                                w.as_ptr(),
                                "0".to_string(),
                                0,
                                opts,
                                &mut count,
                            ))
                        }
                        _ => None,
                    };
                    (ctx.app_name, title, tree, ctx.pid)
                }
                None => {
                    notes.push("no focused application/window found".to_string());
                    (None, None, None, None)
                }
            }
        } else {
            notes.push(permissions::accessibility_hint());
            (None, None, None, None)
        };

    let windows = windows::list_windows(pid);

    // Fall back to the active window's owner for the app name when AX is absent.
    if active_app.is_none() {
        if let Some(w) = windows.iter().find(|w| w.is_active) {
            active_app = Some(w.app.clone());
            if active_window_title.is_none() {
                active_window_title = w.title.clone();
            }
        }
    }

    let screenshot = if opts.include_screenshot {
        match screenshot::capture_main_display(screenshot_dir, screen_size, cursor) {
            Ok(shot) => Some(shot),
            Err(e) => {
                notes.push(format!("screenshot unavailable: {e}"));
                None
            }
        }
    } else {
        None
    };

    Ok(ComputerObservation {
        screen_size,
        active_app,
        active_window_title,
        windows,
        accessibility_tree: tree,
        cursor: cursor.map(|p| Point { x: p.x, y: p.y }),
        screenshot,
        notes,
    })
}

/// Apply a single non-observing action.
pub fn act(action: &ComputerAction) -> Result<ActionOutcome, ComputerUseError> {
    // Input and accessibility actions need the Accessibility grant (synthetic
    // events are dropped silently without it); guard up front for a clear error.
    if requires_accessibility(action) && !permissions::accessibility_trusted() {
        return Err(ComputerUseError::PermissionDenied(
            permissions::accessibility_hint(),
        ));
    }

    match action {
        ComputerAction::Click { x, y, button } => {
            input::click(*x, *y, *button);
            Ok(ActionOutcome::ok(format!(
                "clicked {button:?} at ({x:.0}, {y:.0})"
            )))
        }
        ComputerAction::DoubleClick { x, y } => {
            input::double_click(*x, *y);
            Ok(ActionOutcome::ok(format!(
                "double-clicked at ({x:.0}, {y:.0})"
            )))
        }
        ComputerAction::RightClick { x, y } => {
            input::click(*x, *y, sweet_computer_use_core::MouseButton::Right);
            Ok(ActionOutcome::ok(format!(
                "right-clicked at ({x:.0}, {y:.0})"
            )))
        }
        ComputerAction::MoveCursor { x, y } => {
            input::move_cursor(*x, *y);
            Ok(ActionOutcome::ok(format!(
                "moved cursor to ({x:.0}, {y:.0})"
            )))
        }
        ComputerAction::Scroll { x, y, dx, dy } => {
            input::scroll(*x, *y, *dx, *dy);
            Ok(ActionOutcome::ok(format!(
                "scrolled (dx={dx}, dy={dy}) at ({x:.0}, {y:.0})"
            )))
        }
        ComputerAction::Drag {
            from_x,
            from_y,
            to_x,
            to_y,
        } => {
            input::drag(*from_x, *from_y, *to_x, *to_y);
            Ok(ActionOutcome::ok(format!(
                "dragged ({from_x:.0}, {from_y:.0}) -> ({to_x:.0}, {to_y:.0})"
            )))
        }
        ComputerAction::TypeText { text } => {
            input::type_text(text);
            Ok(ActionOutcome::ok(format!(
                "typed {} characters",
                text.chars().count()
            )))
        }
        ComputerAction::KeyChord { keys } => {
            input::key_chord(keys)?;
            Ok(ActionOutcome::ok(format!("pressed {}", keys.join("+"))))
        }
        ComputerAction::AxPress { element } => {
            accessibility::press(element)?;
            Ok(ActionOutcome::ok(format!("pressed element {element}")))
        }
        ComputerAction::AxSetValue { element, value } => {
            accessibility::set_value(element, value)?;
            Ok(ActionOutcome::ok(format!("set value of element {element}")))
        }
        ComputerAction::OpenApp { name } => {
            open_app(name)?;
            Ok(ActionOutcome::ok(format!("opened {name}")))
        }
        // `Wait` is resolved by the async provider (a non-blocking
        // `tokio::time::sleep`) before it reaches this sync function;
        // observe/screenshot are routed to `observe`. None are real platform
        // actions, so reaching them here is a contract violation.
        ComputerAction::Wait { .. }
        | ComputerAction::Observe { .. }
        | ComputerAction::Screenshot => Err(ComputerUseError::InvalidAction(
            "wait/observe/screenshot are handled outside the platform act path".to_string(),
        )),
    }
}

/// Whether an action drives input or the accessibility API (and thus needs the
/// Accessibility permission).
fn requires_accessibility(action: &ComputerAction) -> bool {
    !matches!(
        action,
        ComputerAction::Wait { .. }
            | ComputerAction::OpenApp { .. }
            | ComputerAction::Observe { .. }
            | ComputerAction::Screenshot
    )
}

/// Launch or focus an application by name via `/usr/bin/open`.
///
/// Deliberately bypasses `sweet-core`'s `CommandRunner`: this tool drives the
/// host desktop GUI (a `Dangerous`, approval-gated action outside the project
/// sandbox by nature), so the project's sandboxed shell policy does not apply.
/// The path is a fixed system binary, not caller-controlled, so there is no
/// injection surface.
fn open_app(name: &str) -> Result<(), ComputerUseError> {
    let status = std::process::Command::new("/usr/bin/open")
        .arg("-a")
        .arg(name)
        .status()
        .map_err(|e| ComputerUseError::Platform(format!("failed to run `open`: {e}")))?;
    if status.success() {
        Ok(())
    } else {
        Err(ComputerUseError::Platform(format!(
            "`open -a {name:?}` exited with {status}"
        )))
    }
}
