// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Structured snapshot of the GUI returned by `observe`, the options that shape
//! it, and the result type for non-observing actions.

use serde::{Deserialize, Serialize};

use crate::action::Point;

/// Dimensions of the main display in the observation's coordinate space:
/// logical points, top-left origin - the same space as element frames, the
/// cursor, and the coordinates `click`/`move`/etc. expect.
///
/// A HiDPI/Retina backend captures the screen in physical pixels (larger than
/// these points), so it resizes the capture to this point size before returning
/// it - see [`Size::screenshot_target`] - keeping the screenshot in the same
/// coordinate space as clicks.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

impl Size {
    /// The pixel dimensions a screenshot should be resized to so it shares this
    /// (point-valued) size's coordinate space, given the raw capture's pixel
    /// dimensions.
    ///
    /// `None` means "leave the capture as-is": either this point size is
    /// unusable (non-positive), or the capture is already no larger than the
    /// target (a non-HiDPI display, where pixels already equal points).
    ///
    /// This deliberately imposes **no upper cap**: the screenshot must match the
    /// display's point size *exactly* for a position read off it to map 1:1 to a
    /// click. A "More Space" display (e.g. 1800 points wide) must stay 1800 px,
    /// not be clamped to something smaller.
    pub fn screenshot_target(self, capture_w: u32, capture_h: u32) -> Option<(u32, u32)> {
        if self.width <= 0.0 || self.height <= 0.0 {
            return None;
        }
        let target_w = (self.width.round() as u32).max(1);
        let target_h = (self.height.round() as u32).max(1);
        if capture_w <= target_w && capture_h <= target_h {
            return None;
        }
        Some((target_w, target_h))
    }
}

/// A rectangle in global display coordinates (top-left origin).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    /// The center point - a sensible click target for the element this frame
    /// describes.
    pub fn center(&self) -> Point {
        Point {
            x: self.x + self.width / 2.0,
            y: self.y + self.height / 2.0,
        }
    }
}

/// A rectangle of pixels to fill, in screenshot pixel coordinates. Used by a
/// backend to draw the cursor marker onto a captured screenshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Fill rectangles for a crosshair marker centered at `(cx, cy)` in an image of
/// `img_w`x`img_h` pixels, clipped to the image bounds.
///
/// The crosshair is four arms with a small central gap, so the exact target
/// pixel stays visible. A backend draws each rectangle (typically with a
/// contrasting outline) to show the model where the cursor currently points, so
/// it can aim *relative to the marker* and verify after moving - far more
/// reliable than estimating absolute coordinates from a single screenshot. Arms
/// that fall entirely outside the image are dropped.
pub fn crosshair_rects(cx: f64, cy: f64, img_w: u32, img_h: u32) -> Vec<PixelRect> {
    const THICK: i64 = 3;
    const ARM: i64 = 22;
    const GAP: i64 = 6;
    let cx = cx.round() as i64;
    let cy = cy.round() as i64;
    let half = THICK / 2;
    // (x, y, w, h) for the up / down / left / right arms.
    let arms = [
        (cx - half, cy - GAP - ARM, THICK, ARM),
        (cx - half, cy + GAP, THICK, ARM),
        (cx - GAP - ARM, cy - half, ARM, THICK),
        (cx + GAP, cy - half, ARM, THICK),
    ];
    arms.iter()
        .filter_map(|&(x, y, w, h)| clip_rect(x, y, w, h, img_w, img_h))
        .collect()
}

/// Clip a rectangle to `[0, img_w) x [0, img_h)`, returning `None` if nothing is
/// left inside the image.
fn clip_rect(x: i64, y: i64, w: i64, h: i64, img_w: u32, img_h: u32) -> Option<PixelRect> {
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + w).min(img_w as i64);
    let y1 = (y + h).min(img_h as i64);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(PixelRect {
        x: x0 as u32,
        y: y0 as u32,
        width: (x1 - x0) as u32,
        height: (y1 - y0) as u32,
    })
}

/// A top-level window currently on screen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowInfo {
    /// Owning application name.
    pub app: String,
    /// Window title, if it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// On-screen bounds.
    pub bounds: Rect,
    /// Whether this window belongs to the frontmost application.
    pub is_active: bool,
    /// Platform window id (macOS `kCGWindowNumber`).
    pub window_id: u32,
}

/// A node in the accessibility tree.
///
/// The [`path`](Self::path) is a slash-joined chain of child indices from the
/// root window (e.g. `"0/2/1"`). It is how the model targets an element for
/// [`AxPress`](crate::ComputerAction::AxPress) /
/// [`AxSetValue`](crate::ComputerAction::AxSetValue): the provider re-walks the
/// tree by those indices to recover the live element handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiNode {
    /// Child-index path from the root window, e.g. `"0/2/1"`.
    pub path: String,
    /// Accessibility role, e.g. `AXButton`, `AXTextField`, `AXWindow`.
    pub role: String,
    /// `AXTitle`, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// `AXDescription` / accessibility label, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// `AXValue` rendered as text (truncated), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// `AXIdentifier`, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identifier: Option<String>,
    /// On-screen frame, if the element reports position and size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame: Option<Rect>,
    /// Whether the element is enabled.
    pub enabled: bool,
    /// Whether the element is focused.
    pub focused: bool,
    /// Available accessibility actions, e.g. `["AXPress"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<String>,
    /// Child nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<UiNode>,
}

/// A captured screenshot. The raw bytes are kept out of serde output (they are
/// large and persisted separately on disk); only metadata round-trips.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Screenshot {
    /// Raw image bytes (PNG). Not serialized - see the struct docs.
    #[serde(skip)]
    pub data: Vec<u8>,
    /// MIME type, e.g. `"image/png"`.
    pub media_type: String,
    /// Filesystem path the image was written to, if it was saved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Debug for Screenshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Screenshot")
            .field("bytes", &self.data.len())
            .field("media_type", &self.media_type)
            .field("path", &self.path)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

/// A textual snapshot of the GUI, returned by
/// [`observe`](crate::ComputerUseProvider::observe).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComputerObservation {
    pub screen_size: Size,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_app: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_window_title: Option<String>,
    #[serde(default)]
    pub windows: Vec<WindowInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accessibility_tree: Option<UiNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<Point>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screenshot: Option<Screenshot>,
    /// Non-fatal remarks surfaced to the model, e.g.
    /// `"screenshot unavailable: screen-recording permission not granted"`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// Options shaping an [`observe`](crate::ComputerUseProvider::observe) call.
#[derive(Debug, Clone)]
pub struct ObserveOptions {
    /// Capture a screenshot to disk.
    pub include_screenshot: bool,
    /// Walk and include the accessibility tree.
    pub include_tree: bool,
    /// Maximum accessibility-tree depth to descend.
    pub max_depth: usize,
    /// Maximum number of accessibility nodes to emit (a budget guarding against
    /// pathologically large trees blowing the model's context).
    pub max_nodes: usize,
}

impl Default for ObserveOptions {
    fn default() -> Self {
        Self {
            include_screenshot: true,
            include_tree: true,
            max_depth: 16,
            max_nodes: 250,
        }
    }
}

/// The result of a non-observing action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// Whether the action was applied successfully.
    pub ok: bool,
    /// Human-readable detail describing what happened.
    pub detail: String,
}

impl ActionOutcome {
    /// A successful outcome with a description.
    pub fn ok(detail: impl Into<String>) -> Self {
        Self {
            ok: true,
            detail: detail.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sz(w: f64, h: f64) -> Size {
        Size {
            width: w,
            height: h,
        }
    }

    #[test]
    fn screenshot_target_downscales_retina_default() {
        // 14" MBP default: 1512x982 points, captured at 2x (3024x1964).
        assert_eq!(
            sz(1512.0, 982.0).screenshot_target(3024, 1964),
            Some((1512, 982))
        );
    }

    #[test]
    fn screenshot_target_does_not_clamp_more_space() {
        // 14" MBP "More Space": 1800x1169 points - must stay 1800 px wide, the
        // exact regression the old 1568 long-edge cap caused.
        assert_eq!(
            sz(1800.0, 1169.0).screenshot_target(3600, 2338),
            Some((1800, 1169))
        );
    }

    #[test]
    fn screenshot_target_skips_when_capture_not_larger() {
        // Non-HiDPI display: pixels already equal points, nothing to resize.
        assert_eq!(sz(1920.0, 1080.0).screenshot_target(1920, 1080), None);
        assert_eq!(sz(1920.0, 1080.0).screenshot_target(1900, 1070), None);
    }

    #[test]
    fn screenshot_target_skips_degenerate_point_size() {
        assert_eq!(sz(0.0, 0.0).screenshot_target(3024, 1964), None);
    }

    fn in_bounds(r: &PixelRect, w: u32, h: u32) -> bool {
        r.width > 0 && r.height > 0 && r.x + r.width <= w && r.y + r.height <= h
    }

    #[test]
    fn crosshair_centered_has_four_in_bounds_arms() {
        let rects = crosshair_rects(900.0, 584.0, 1800, 1169);
        assert_eq!(rects.len(), 4);
        assert!(rects.iter().all(|r| in_bounds(r, 1800, 1169)));
    }

    #[test]
    fn crosshair_at_corner_drops_off_screen_arms_and_clips() {
        // At the top-left corner the up and left arms fall off-screen.
        let rects = crosshair_rects(0.0, 0.0, 1800, 1169);
        assert!(rects.len() < 4);
        assert!(rects.iter().all(|r| in_bounds(r, 1800, 1169)));
    }

    #[test]
    fn crosshair_fully_off_screen_is_empty() {
        assert!(crosshair_rects(5000.0, 5000.0, 1800, 1169).is_empty());
    }
}
