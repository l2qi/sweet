// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! The single model-facing `computer` tool that bridges actions to a provider.

use std::sync::{Arc, Mutex};

use sweet_core::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::tool::{execution_error, ToolError, ToolHandler, ToolOutput, ToolSpec};

use crate::action::ComputerAction;
use crate::observation::{ComputerObservation, ObserveOptions, Size};
use crate::provider::SharedProvider;
use crate::render::{render_observation, render_outcome};

/// The protocol name of the computer-use tool.
pub const COMPUTER_TOOL_NAME: &str = "computer";

/// The coordinate convention a backend's model emits, used to map model
/// coordinates to display pixels deterministically (the harness owns the
/// scaling; the model never does coordinate arithmetic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateSpace {
    /// Coordinates are already absolute display pixels - used as-is.
    Absolute,
    /// Coordinates are normalized to `[0, grid]` on each axis (a common
    /// vision-model grounding convention; e.g. `grid = 1000`). The tool maps
    /// them to pixels with the current display size: `px = coord / grid * dim`.
    Normalized { grid: f64 },
}

impl CoordinateSpace {
    /// The per-axis multiplier that turns a model coordinate into a display
    /// pixel, given the display's point size.
    fn scale(self, screen: Size) -> (f64, f64) {
        match self {
            CoordinateSpace::Absolute => (1.0, 1.0),
            CoordinateSpace::Normalized { grid } => (screen.width / grid, screen.height / grid),
        }
    }
}

const DESCRIPTION: &str = "\
Observe and control the local macOS desktop GUI: read the screen's accessibility \
state and perform bounded mouse/keyboard actions. Use this only for tasks that \
genuinely require the GUI (driving a desktop app, a browser, or visually \
verifying running software) - prefer file, shell, and search tools for ordinary \
coding work.

Always `observe` first. The observation lists the active app, on-screen windows, \
and the accessibility tree of the focused window. Each tree element shows a \
path like [0/2/1], its role, title/label/value, and on-screen frame \
@(x,y w x h). Act on one element at a time, then `observe` again to confirm.

STRONGLY PREFER `ax_press`/`ax_set_value` (targeting an element by its [path]) \
over raw coordinate clicks whenever the element appears in the tree: they hit \
the element exactly, are robust to layout shifts, and need no coordinates at \
all. To fill a text field, `ax_set_value` it. Reach for `click`/`move` only \
when there is no usable accessibility element (e.g. canvas or web content).

For coordinate actions, point at where your target is in the screenshot - the \
tool maps your coordinates to the screen for you, so you do not need to scale or \
convert anything. The current cursor is drawn on each screenshot as a magenta \
crosshair (and reported as `Cursor: (x, y)`): aim relative to the crosshair, \
`move` there, then `observe` again to see where you landed and nudge before you \
`click`. This observe -> move -> verify loop is the reliable way to hit a target.

Set `action` to one of: \
observe { include_screenshot? } | screenshot | click { x, y, button? } | \
double_click { x, y } | right_click { x, y } | move { x, y } | \
scroll { x, y, dx?, dy? } | drag { from_x, from_y, to_x, to_y } | \
type_text { text } | key_chord { keys: [..] } | ax_press { element } | \
ax_set_value { element, value } | wait { millis } | open_app { name }.";

/// Build the `computer` tool backed by `provider`, interpreting model
/// coordinates according to `coordinate_space`.
///
/// Observe-style actions are routed to the provider's `observe`; all other
/// actions to `act`. Backend errors surface as tool-execution errors with the
/// backend's own (actionable) message.
pub fn computer_use_tool(provider: SharedProvider, coordinate_space: CoordinateSpace) -> ToolSpec {
    ToolSpec::new(
        COMPUTER_TOOL_NAME,
        DESCRIPTION,
        schema(),
        ComputerUseHandler {
            provider,
            coordinate_space,
            screen: Arc::new(Mutex::new(None)),
        },
    )
    .with_risk(ToolRisk::Dangerous)
}

struct ComputerUseHandler {
    provider: SharedProvider,
    coordinate_space: CoordinateSpace,
    /// Last observed display size, used to map model coordinates to pixels
    /// without an extra round-trip. Refreshed on every observe.
    screen: Arc<Mutex<Option<Size>>>,
}

#[async_trait]
impl ToolHandler for ComputerUseHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        // Text-only path: the rich output's text (the image, if any, is dropped).
        Ok(self.call_rich(args).await?.text_content())
    }

    async fn call_rich(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        // `?` converts a parse failure into ToolError::InvalidArgs.
        let action: ComputerAction = serde_json::from_value(args)?;
        match action {
            ComputerAction::Observe { include_screenshot } => {
                let opts = ObserveOptions {
                    include_screenshot,
                    ..ObserveOptions::default()
                };
                let obs = self.observe(&opts).await?;
                Ok(observation_output(&obs))
            }
            ComputerAction::Screenshot => {
                let opts = ObserveOptions {
                    include_screenshot: true,
                    include_tree: false,
                    ..ObserveOptions::default()
                };
                let obs = self.observe(&opts).await?;
                Ok(observation_output(&obs))
            }
            other => {
                let action = self.to_pixels(other).await?;
                let outcome = self.provider.act(&action).await.map_err(execution_error)?;
                Ok(ToolOutput::text(render_outcome(&outcome)))
            }
        }
    }
}

impl ComputerUseHandler {
    /// Observe via the provider, caching the reported display size for later
    /// coordinate mapping.
    async fn observe(&self, opts: &ObserveOptions) -> Result<ComputerObservation, ToolError> {
        let obs = self.provider.observe(opts).await.map_err(execution_error)?;
        if obs.screen_size.width > 0.0 && obs.screen_size.height > 0.0 {
            *self.screen.lock().unwrap() = Some(obs.screen_size);
        }
        Ok(obs)
    }

    /// The current display size - cached from the last observe, or fetched once
    /// if no observe has happened yet this session.
    async fn screen_size(&self) -> Result<Size, ToolError> {
        if let Some(size) = *self.screen.lock().unwrap() {
            return Ok(size);
        }
        let opts = ObserveOptions {
            include_screenshot: false,
            include_tree: false,
            ..ObserveOptions::default()
        };
        Ok(self.observe(&opts).await?.screen_size)
    }

    /// Map a model-supplied action's coordinates to display pixels (per
    /// [`CoordinateSpace`]) and clamp them on-screen. Non-coordinate actions are
    /// returned unchanged without touching the display.
    async fn to_pixels(&self, action: ComputerAction) -> Result<ComputerAction, ToolError> {
        // Absolute space with nothing to clamp against would be a no-op anyway;
        // only pay for a screen-size lookup when it can change the action.
        if matches!(self.coordinate_space, CoordinateSpace::Absolute) {
            return Ok(action);
        }
        let screen = self.screen_size().await?;
        let (sx, sy) = self.coordinate_space.scale(screen);
        Ok(action.scaled(sx, sy).clamped(
            (screen.width - 1.0).max(0.0),
            (screen.height - 1.0).max(0.0),
        ))
    }
}

/// Render an observation to text and attach its screenshot (if captured) as an
/// image block, so vision-capable models receive the pixels alongside the
/// accessibility text. On text-only protocols the image is dropped downstream.
fn observation_output(obs: &ComputerObservation) -> ToolOutput {
    let mut out = ToolOutput::text(render_observation(obs));
    if let Some(shot) = &obs.screenshot {
        if !shot.data.is_empty() {
            out = out.with_image(shot.data.clone(), shot.media_type.clone());
        }
    }
    out
}

/// JSON Schema for the tool input. Kept permissive (only `action` is required)
/// because the parameter set is action-dependent; the detailed contract lives
/// in [`DESCRIPTION`]. Validation happens during `from_value` against
/// [`ComputerAction`].
fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "observe", "screenshot", "click", "double_click", "right_click",
                    "move", "scroll", "drag", "type_text", "key_chord",
                    "ax_press", "ax_set_value", "wait", "open_app"
                ],
                "description": "Which GUI action to perform."
            },
            "x": { "type": "number", "description": "Target X (top-left origin)." },
            "y": { "type": "number", "description": "Target Y (top-left origin)." },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button for `click` (default left)."
            },
            "dx": { "type": "number", "description": "Horizontal scroll in wheel lines." },
            "dy": { "type": "number", "description": "Vertical scroll in wheel lines (positive scrolls up)." },
            "from_x": { "type": "number" },
            "from_y": { "type": "number" },
            "to_x": { "type": "number" },
            "to_y": { "type": "number" },
            "text": { "type": "string", "description": "Text to type for `type_text`." },
            "keys": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Key chord for `key_chord`, e.g. [\"cmd\", \"l\"]."
            },
            "element": {
                "type": "string",
                "description": "Accessibility element path (e.g. \"0/2/1\") for `ax_press`/`ax_set_value`."
            },
            "value": { "type": "string", "description": "New value for `ax_set_value`." },
            "millis": { "type": "integer", "minimum": 0, "description": "Pause length for `wait`." },
            "name": { "type": "string", "description": "Application name for `open_app`." },
            "include_screenshot": { "type": "boolean", "description": "Capture a screenshot during `observe` (default true)." }
        },
        "required": ["action"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::MouseButton;
    use crate::observation::{ActionOutcome, Size};
    use crate::provider::{ComputerUseError, ComputerUseProvider};
    use std::sync::Arc;
    use std::sync::Mutex;

    #[derive(Default)]
    struct FakeProvider {
        last_action: Mutex<Option<ComputerAction>>,
        last_include_screenshot: Mutex<Option<bool>>,
    }

    #[async_trait]
    impl ComputerUseProvider for FakeProvider {
        async fn observe(
            &self,
            opts: &ObserveOptions,
        ) -> Result<ComputerObservation, ComputerUseError> {
            *self.last_include_screenshot.lock().unwrap() = Some(opts.include_screenshot);
            Ok(ComputerObservation {
                screen_size: Size {
                    width: 1800.0,
                    height: 1169.0,
                },
                active_app: Some("TestApp".into()),
                ..Default::default()
            })
        }

        async fn act(&self, action: &ComputerAction) -> Result<ActionOutcome, ComputerUseError> {
            *self.last_action.lock().unwrap() = Some(action.clone());
            Ok(ActionOutcome::ok("did it"))
        }

        fn platform(&self) -> &'static str {
            "fake"
        }
    }

    #[tokio::test]
    async fn observe_is_routed_to_observe() {
        let provider = Arc::new(FakeProvider::default());
        let tool = computer_use_tool(provider.clone(), CoordinateSpace::Absolute);
        let out = tool
            .call(serde_json::json!({"action": "observe", "include_screenshot": false}))
            .await
            .unwrap();
        assert!(out.contains("Active app: TestApp"));
        assert_eq!(
            *provider.last_include_screenshot.lock().unwrap(),
            Some(false)
        );
        assert!(provider.last_action.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn absolute_click_is_passed_through_unchanged() {
        let provider = Arc::new(FakeProvider::default());
        let tool = computer_use_tool(provider.clone(), CoordinateSpace::Absolute);
        let out = tool
            .call(serde_json::json!({"action": "click", "x": 5.0, "y": 6.0}))
            .await
            .unwrap();
        assert_eq!(out, "did it");
        assert_eq!(
            *provider.last_action.lock().unwrap(),
            Some(ComputerAction::Click {
                x: 5.0,
                y: 6.0,
                button: MouseButton::Left
            })
        );
    }

    #[tokio::test]
    async fn normalized_click_is_mapped_to_pixels() {
        // 1800x1169 screen, [0,1000] normalized grid: the model's read of the
        // Filter box at normalized (68, 256) maps to pixel (122, 299) - the bug
        // from the field report, now fixed deterministically.
        let provider = Arc::new(FakeProvider::default());
        let tool = computer_use_tool(
            provider.clone(),
            CoordinateSpace::Normalized { grid: 1000.0 },
        );
        // Observe first so the screen size is cached (mirrors real usage).
        tool.call(serde_json::json!({"action": "observe"}))
            .await
            .unwrap();
        tool.call(serde_json::json!({"action": "click", "x": 68, "y": 256}))
            .await
            .unwrap();
        let recorded = provider.last_action.lock().unwrap().clone().unwrap();
        match recorded {
            ComputerAction::Click { x, y, .. } => {
                assert!((x - 122.4).abs() < 0.5, "x was {x}");
                assert!((y - 299.3).abs() < 0.5, "y was {y}");
            }
            other => panic!("expected click, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn normalized_mapping_fetches_screen_size_without_a_prior_observe() {
        // No observe first: the handler must fetch the screen size on demand.
        let provider = Arc::new(FakeProvider::default());
        let tool = computer_use_tool(
            provider.clone(),
            CoordinateSpace::Normalized { grid: 1000.0 },
        );
        tool.call(serde_json::json!({"action": "click", "x": 500, "y": 500}))
            .await
            .unwrap();
        let recorded = provider.last_action.lock().unwrap().clone().unwrap();
        match recorded {
            ComputerAction::Click { x, y, .. } => {
                assert!((x - 900.0).abs() < 0.5, "x was {x}");
                assert!((y - 584.5).abs() < 0.5, "y was {y}");
            }
            other => panic!("expected click, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn off_screen_coordinates_are_clamped_on_screen() {
        // A negative model coordinate (the field-report failure) clamps to 0
        // rather than firing an off-screen click.
        let provider = Arc::new(FakeProvider::default());
        let tool = computer_use_tool(
            provider.clone(),
            CoordinateSpace::Normalized { grid: 1000.0 },
        );
        tool.call(serde_json::json!({"action": "click", "x": -711, "y": 5000}))
            .await
            .unwrap();
        let recorded = provider.last_action.lock().unwrap().clone().unwrap();
        match recorded {
            ComputerAction::Click { x, y, .. } => {
                assert_eq!(x, 0.0);
                assert_eq!(y, 1168.0); // clamped to height - 1
            }
            other => panic!("expected click, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bad_args_are_invalid_args_error() {
        let provider = Arc::new(FakeProvider::default());
        let tool = computer_use_tool(provider, CoordinateSpace::Absolute);
        let err = tool
            .call(serde_json::json!({"action": "nonsense"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs(_)));
    }
}
