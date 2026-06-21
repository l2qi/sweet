// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Rendering of observations and outcomes into the compact text a model reads.
//!
//! The accessibility tree is the model's primary signal, so the format is tuned
//! to be dense but legible: one element per line, indented by depth, with the
//! targeting path first (`[0/2/1]`), then role, then whichever of
//! title/label/value/frame/actions are present.

use std::fmt::Write as _;

use crate::observation::{ActionOutcome, ComputerObservation, UiNode};

/// Maximum characters of an element's value to show inline before truncating.
const MAX_VALUE_LEN: usize = 120;

/// Render an [`ActionOutcome`] as the tool's text reply.
pub fn render_outcome(outcome: &ActionOutcome) -> String {
    if outcome.ok {
        outcome.detail.clone()
    } else {
        format!("Action did not apply: {}", outcome.detail)
    }
}

/// Render a [`ComputerObservation`] as the tool's text reply.
pub fn render_observation(obs: &ComputerObservation) -> String {
    let mut out = String::new();

    let _ = writeln!(
        out,
        "Screen: {:.0}x{:.0}",
        obs.screen_size.width, obs.screen_size.height
    );
    if let Some(app) = &obs.active_app {
        let _ = writeln!(out, "Active app: {app}");
    }
    if let Some(title) = &obs.active_window_title {
        let _ = writeln!(out, "Active window: {title:?}");
    }
    if let Some(c) = &obs.cursor {
        let _ = writeln!(out, "Cursor: ({:.0}, {:.0})", c.x, c.y);
    }
    if let Some(shot) = &obs.screenshot {
        match &shot.path {
            Some(p) => {
                let _ = writeln!(out, "Screenshot ({}x{}): {p}", shot.width, shot.height);
            }
            None => {
                let _ = writeln!(out, "Screenshot ({}x{}) captured", shot.width, shot.height);
            }
        }
    }
    for note in &obs.notes {
        let _ = writeln!(out, "Note: {note}");
    }

    if !obs.windows.is_empty() {
        let _ = writeln!(out, "\nWindows ({}):", obs.windows.len());
        for w in &obs.windows {
            let title = w.title.as_deref().unwrap_or("");
            let active = if w.is_active { " (active)" } else { "" };
            let _ = writeln!(
                out,
                "  - {} {title:?} [{:.0}x{:.0} @ ({:.0},{:.0})]{active}",
                w.app, w.bounds.width, w.bounds.height, w.bounds.x, w.bounds.y
            );
        }
    }

    match &obs.accessibility_tree {
        Some(root) => {
            let _ = writeln!(out, "\nAccessibility tree (focused window):");
            let mut budget = usize::MAX;
            render_node(&mut out, root, 0, &mut budget);
        }
        None => {
            let _ = writeln!(
                out,
                "\nAccessibility tree: unavailable (no focused window or accessibility data)."
            );
        }
    }

    out
}

/// Append one node and its subtree. `budget` bounds total nodes; rendering stops
/// (with a marker) once it is exhausted. The provider already prunes by depth
/// and node count, so this is a defensive last line.
fn render_node(out: &mut String, node: &UiNode, depth: usize, budget: &mut usize) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;

    let indent = "  ".repeat(depth);
    let mut line = format!("{indent}[{}] {}", node.path, node.role);

    if let Some(t) = &node.title {
        if !t.is_empty() {
            let _ = write!(line, " {t:?}");
        }
    }
    if let Some(l) = &node.label {
        if !l.is_empty() && node.title.as_deref() != Some(l.as_str()) {
            let _ = write!(line, " label={l:?}");
        }
    }
    if let Some(v) = &node.value {
        if !v.is_empty() {
            let _ = write!(line, " value={:?}", truncate(v));
        }
    }
    if let Some(id) = &node.identifier {
        if !id.is_empty() {
            let _ = write!(line, " id={id:?}");
        }
    }
    if let Some(f) = &node.frame {
        let _ = write!(
            line,
            " @({:.0},{:.0} {:.0}x{:.0})",
            f.x, f.y, f.width, f.height
        );
    }
    if !node.enabled {
        line.push_str(" disabled");
    }
    if node.focused {
        line.push_str(" focused");
    }
    if !node.actions.is_empty() {
        let _ = write!(line, " actions=[{}]", node.actions.join(","));
    }

    out.push_str(&line);
    out.push('\n');

    for child in &node.children {
        render_node(out, child, depth + 1, budget);
    }
}

fn truncate(s: &str) -> String {
    if s.chars().count() <= MAX_VALUE_LEN {
        return s.to_string();
    }
    let mut t: String = s.chars().take(MAX_VALUE_LEN).collect();
    t.push('…');
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{Rect, Size};

    fn leaf(path: &str, role: &str) -> UiNode {
        UiNode {
            path: path.into(),
            role: role.into(),
            title: None,
            label: None,
            value: None,
            identifier: None,
            frame: None,
            enabled: true,
            focused: false,
            actions: vec![],
            children: vec![],
        }
    }

    #[test]
    fn renders_app_window_and_tree() {
        let obs = ComputerObservation {
            screen_size: Size {
                width: 1920.0,
                height: 1080.0,
            },
            active_app: Some("Safari".into()),
            active_window_title: Some("Start".into()),
            accessibility_tree: Some(UiNode {
                actions: vec!["AXPress".into()],
                frame: Some(Rect {
                    x: 10.0,
                    y: 20.0,
                    width: 30.0,
                    height: 40.0,
                }),
                title: Some("Back".into()),
                ..leaf("0", "AXButton")
            }),
            ..Default::default()
        };
        let text = render_observation(&obs);
        assert!(text.contains("Active app: Safari"));
        assert!(text.contains("1920x1080"));
        assert!(text.contains("[0] AXButton \"Back\""));
        assert!(text.contains("actions=[AXPress]"));
        assert!(text.contains("@(10,20 30x40)"));
    }

    #[test]
    fn truncates_long_values() {
        let long = "x".repeat(500);
        let mut node = leaf("0", "AXTextArea");
        node.value = Some(long);
        let mut out = String::new();
        let mut budget = usize::MAX;
        render_node(&mut out, &node, 0, &mut budget);
        assert!(out.contains('…'));
        assert!(out.len() < 300);
    }

    #[test]
    fn node_budget_caps_output() {
        let mut root = leaf("0", "AXWindow");
        root.children = (0..100).map(|i| leaf(&format!("0/{i}"), "AXRow")).collect();
        let mut out = String::new();
        let mut budget = 5;
        render_node(&mut out, &root, 0, &mut budget);
        // root + 4 children = 5 lines, no more.
        assert_eq!(out.lines().count(), 5);
    }

    #[test]
    fn outcome_renders_detail() {
        assert_eq!(
            render_outcome(&ActionOutcome::ok("clicked (10, 20)")),
            "clicked (10, 20)"
        );
        let failed = ActionOutcome {
            ok: false,
            detail: "no element".into(),
        };
        assert!(render_outcome(&failed).starts_with("Action did not apply"));
    }
}
