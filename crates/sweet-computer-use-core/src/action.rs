// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! The bounded set of GUI actions a model can request through the `computer` tool.

use serde::{Deserialize, Serialize};

/// A point in global display coordinates.
///
/// The origin is the top-left of the main display - the same space macOS
/// accessibility frames, the cursor position, and synthetic mouse events all
/// use. A frame reported by [`observe`](crate::ComputerUseProvider::observe)
/// can therefore be clicked directly, with no coordinate conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Which mouse button a click targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    #[default]
    Left,
    Right,
    Middle,
}

/// A single bounded GUI action the model can request.
///
/// This is the wire shape the model produces: an object with an `action`
/// discriminator plus action-specific fields, e.g.
/// `{"action": "click", "x": 12.0, "y": 40.0}` or
/// `{"action": "key_chord", "keys": ["cmd", "l"]}`. The variants are
/// deliberately small and explicit so the action space is auditable.
///
/// [`Observe`](Self::Observe) and [`Screenshot`](Self::Screenshot) are handled
/// by the tool itself (routed to the provider's `observe`); every other variant
/// is forwarded to [`act`](crate::ComputerUseProvider::act).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ComputerAction {
    /// Report the current GUI state: active app/window, on-screen windows, and
    /// the accessibility tree of the focused window.
    Observe {
        /// Also capture a screenshot to disk and report its path. Defaults to
        /// `true`; set `false` to skip capture (e.g. when permission is absent
        /// or the model only needs structural data).
        #[serde(default = "default_true")]
        include_screenshot: bool,
    },
    /// Capture a screenshot to disk and report its path (no accessibility tree).
    Screenshot,
    /// Click at a point with the given button (defaults to left).
    Click {
        #[serde(deserialize_with = "de_coord")]
        x: f64,
        #[serde(deserialize_with = "de_coord")]
        y: f64,
        #[serde(default)]
        button: MouseButton,
    },
    /// Double-click (left button) at a point.
    DoubleClick {
        #[serde(deserialize_with = "de_coord")]
        x: f64,
        #[serde(deserialize_with = "de_coord")]
        y: f64,
    },
    /// Right-click (secondary button) at a point.
    RightClick {
        #[serde(deserialize_with = "de_coord")]
        x: f64,
        #[serde(deserialize_with = "de_coord")]
        y: f64,
    },
    /// Move the cursor to a point without pressing any button.
    #[serde(rename = "move")]
    MoveCursor {
        #[serde(deserialize_with = "de_coord")]
        x: f64,
        #[serde(deserialize_with = "de_coord")]
        y: f64,
    },
    /// Scroll at a point. `dy` is vertical (positive scrolls up), `dx` is
    /// horizontal, both in **wheel lines**: integer granularity, rounded to the
    /// nearest whole line (a fractional delta like `2.9` scrolls three lines).
    Scroll {
        #[serde(deserialize_with = "de_coord")]
        x: f64,
        #[serde(deserialize_with = "de_coord")]
        y: f64,
        #[serde(default, deserialize_with = "de_coord")]
        dx: f64,
        #[serde(default, deserialize_with = "de_coord")]
        dy: f64,
    },
    /// Press at `from`, drag to `to`, and release (left button).
    Drag {
        #[serde(deserialize_with = "de_coord")]
        from_x: f64,
        #[serde(deserialize_with = "de_coord")]
        from_y: f64,
        #[serde(deserialize_with = "de_coord")]
        to_x: f64,
        #[serde(deserialize_with = "de_coord")]
        to_y: f64,
    },
    /// Type a Unicode string into the focused element as keystrokes.
    TypeText { text: String },
    /// Press a key chord, e.g. `["cmd", "l"]` or `["cmd", "shift", "t"]`.
    /// Modifier names (`cmd`/`command`, `shift`, `alt`/`option`, `ctrl`/`control`,
    /// `fn`) are applied as flags to the remaining key(s).
    KeyChord { keys: Vec<String> },
    /// Perform the `AXPress` action on the accessibility element identified by
    /// `element` - the bracketed path shown in an `observe` tree (e.g. `0/1/3`).
    /// Preferred over a coordinate click when the element exposes it, since it
    /// is robust to layout shifts and off-screen positioning.
    AxPress { element: String },
    /// Set the `AXValue` of an accessibility element (e.g. fill a text field)
    /// identified by its `observe`-tree path.
    AxSetValue { element: String, value: String },
    /// Pause for `millis` milliseconds (e.g. to let a UI settle).
    Wait { millis: u64 },
    /// Launch or focus an application by name (e.g. `"Safari"`).
    OpenApp { name: String },
}

impl ComputerAction {
    /// Map this action's coordinate fields from the model's coordinate space to
    /// display pixels, applying the per-axis scale `(sx, sy)`.
    ///
    /// Only positional fields are scaled; scroll deltas (`dx`/`dy`, in wheel
    /// lines) and non-coordinate actions are returned unchanged.
    pub(crate) fn scaled(self, sx: f64, sy: f64) -> ComputerAction {
        use ComputerAction::*;
        match self {
            Click { x, y, button } => Click {
                x: x * sx,
                y: y * sy,
                button,
            },
            DoubleClick { x, y } => DoubleClick {
                x: x * sx,
                y: y * sy,
            },
            RightClick { x, y } => RightClick {
                x: x * sx,
                y: y * sy,
            },
            MoveCursor { x, y } => MoveCursor {
                x: x * sx,
                y: y * sy,
            },
            Scroll { x, y, dx, dy } => Scroll {
                x: x * sx,
                y: y * sy,
                dx,
                dy,
            },
            Drag {
                from_x,
                from_y,
                to_x,
                to_y,
            } => Drag {
                from_x: from_x * sx,
                from_y: from_y * sy,
                to_x: to_x * sx,
                to_y: to_y * sy,
            },
            other => other,
        }
    }

    /// Clamp this action's coordinate fields into `[0, max_x] x [0, max_y]`, so a
    /// model that points off-screen (e.g. a negative or oversized coordinate)
    /// still lands on a visible pixel rather than a no-op. Non-coordinate
    /// actions are returned unchanged.
    pub(crate) fn clamped(self, max_x: f64, max_y: f64) -> ComputerAction {
        let cx = |v: f64| v.clamp(0.0, max_x);
        let cy = |v: f64| v.clamp(0.0, max_y);
        use ComputerAction::*;
        match self {
            Click { x, y, button } => Click {
                x: cx(x),
                y: cy(y),
                button,
            },
            DoubleClick { x, y } => DoubleClick { x: cx(x), y: cy(y) },
            RightClick { x, y } => RightClick { x: cx(x), y: cy(y) },
            MoveCursor { x, y } => MoveCursor { x: cx(x), y: cy(y) },
            Scroll { x, y, dx, dy } => Scroll {
                x: cx(x),
                y: cy(y),
                dx,
                dy,
            },
            Drag {
                from_x,
                from_y,
                to_x,
                to_y,
            } => Drag {
                from_x: cx(from_x),
                from_y: cy(from_y),
                to_x: cx(to_x),
                to_y: cy(to_y),
            },
            other => other,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Deserialize an `f64` coordinate, tolerating a model that emits the number as
/// a JSON string (e.g. `{"x": "583"}`) - a common quirk of vision models. A
/// plain number is taken as-is; a string is trimmed and parsed.
fn de_coord<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NumOrStr {
        Num(f64),
        Str(String),
    }
    match NumOrStr::deserialize(deserializer)? {
        NumOrStr::Num(n) => Ok(n),
        NumOrStr::Str(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| serde::de::Error::custom(format!("expected a number, got string {s:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn click_parses_with_default_button() {
        let a: ComputerAction =
            serde_json::from_value(serde_json::json!({"action": "click", "x": 1.0, "y": 2.0}))
                .unwrap();
        assert_eq!(
            a,
            ComputerAction::Click {
                x: 1.0,
                y: 2.0,
                button: MouseButton::Left
            }
        );
    }

    #[test]
    fn observe_defaults_to_capturing_a_screenshot() {
        let a: ComputerAction =
            serde_json::from_value(serde_json::json!({"action": "observe"})).unwrap();
        assert_eq!(
            a,
            ComputerAction::Observe {
                include_screenshot: true
            }
        );
    }

    #[test]
    fn move_uses_the_short_tag() {
        let a: ComputerAction =
            serde_json::from_value(serde_json::json!({"action": "move", "x": 3.0, "y": 4.0}))
                .unwrap();
        assert_eq!(a, ComputerAction::MoveCursor { x: 3.0, y: 4.0 });
    }

    #[test]
    fn key_chord_round_trips() {
        let a = ComputerAction::KeyChord {
            keys: vec!["cmd".into(), "l".into()],
        };
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v["action"], "key_chord");
        assert_eq!(serde_json::from_value::<ComputerAction>(v).unwrap(), a);
    }

    #[test]
    fn click_parses_coords_given_as_strings() {
        // Vision models sometimes emit coordinates as JSON strings.
        let a: ComputerAction =
            serde_json::from_value(serde_json::json!({"action": "click", "x": "583", "y": "193"}))
                .unwrap();
        assert_eq!(
            a,
            ComputerAction::Click {
                x: 583.0,
                y: 193.0,
                button: MouseButton::Left
            }
        );
    }

    #[test]
    fn unknown_action_is_rejected() {
        let r: Result<ComputerAction, _> =
            serde_json::from_value(serde_json::json!({"action": "explode"}));
        assert!(r.is_err());
    }

    #[test]
    fn scaled_maps_positional_fields_but_not_scroll_deltas() {
        let click = ComputerAction::Click {
            x: 100.0,
            y: 200.0,
            button: MouseButton::Left,
        }
        .scaled(1.8, 1.169);
        assert_eq!(
            click,
            ComputerAction::Click {
                x: 180.0,
                y: 233.8,
                button: MouseButton::Left
            }
        );

        let scroll = ComputerAction::Scroll {
            x: 10.0,
            y: 20.0,
            dx: 3.0,
            dy: -5.0,
        }
        .scaled(2.0, 0.5);
        assert_eq!(
            scroll,
            ComputerAction::Scroll {
                x: 20.0,
                y: 10.0,
                dx: 3.0,
                dy: -5.0
            }
        );
    }

    #[test]
    fn scaled_leaves_non_coordinate_actions_unchanged() {
        let t = ComputerAction::TypeText { text: "hi".into() };
        assert_eq!(t.clone().scaled(2.0, 2.0), t);
    }
}
