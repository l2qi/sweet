// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Manual probe for the macOS computer-use backend.
//!
//! Drives [`MacComputerUse`] directly - no model, no app, no agent loop - so
//! you can confirm the Accessibility / Screen-Recording permissions and the FFI
//! work in isolation, separate from whether a model decides to call the tool.
//!
//! Run on a Mac, in a local GUI session, from a terminal that has been granted
//! Accessibility (and Screen Recording, for screenshots):
//!
//! ```text
//! cargo run -p sweet-computer-use-macos --example probe -- observe
//! cargo run -p sweet-computer-use-macos --example probe -- screenshot
//! cargo run -p sweet-computer-use-macos --example probe -- open_app Calculator
//! cargo run -p sweet-computer-use-macos --example probe -- click 200 120
//! cargo run -p sweet-computer-use-macos --example probe -- type_text "hello"
//! cargo run -p sweet-computer-use-macos --example probe -- key_chord cmd l
//! cargo run -p sweet-computer-use-macos --example probe -- ax_press 0/2/1
//! ```
//!
//! With no arguments it observes. On non-macOS targets every action returns
//! `Unsupported`.

use std::env;

use sweet_computer_use_core::{
    render_observation, render_outcome, ComputerAction, ComputerUseProvider, MouseButton,
    ObserveOptions,
};
use sweet_computer_use_macos::MacComputerUse;

const USAGE: &str = "actions: observe | screenshot | click X Y | double_click X Y | \
right_click X Y | move X Y | scroll X Y DX DY | drag X1 Y1 X2 Y2 | type_text TEXT... | \
key_chord KEY... | ax_press PATH | ax_set_value PATH VALUE... | wait MS | open_app NAME...";

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let dir = env::temp_dir().join("sweet-probe-screenshots");
    eprintln!("probe: screenshots -> {}", dir.display());
    let provider = MacComputerUse::new(dir);

    let action = match parse_action(&args) {
        Ok(action) => action,
        Err(msg) => {
            eprintln!("{msg}\n{USAGE}");
            std::process::exit(2);
        }
    };

    match action {
        ComputerAction::Observe { .. } | ComputerAction::Screenshot => {
            match provider.observe(&ObserveOptions::default()).await {
                Ok(observation) => println!("{}", render_observation(&observation)),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        }
        other => match provider.act(&other).await {
            Ok(outcome) => println!("{}", render_outcome(&outcome)),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
    }
}

/// Parse positional arguments into a [`ComputerAction`]. Defaults to `observe`.
fn parse_action(args: &[String]) -> Result<ComputerAction, String> {
    let name = args.first().map(String::as_str).unwrap_or("observe");
    let rest = if args.is_empty() { args } else { &args[1..] };

    let num = |i: usize| -> Result<f64, String> {
        rest.get(i)
            .ok_or_else(|| format!("`{name}` needs a number at position {}", i + 1))?
            .parse::<f64>()
            .map_err(|e| format!("`{name}`: bad number: {e}"))
    };
    let first = || -> Result<String, String> {
        rest.first()
            .cloned()
            .ok_or_else(|| format!("`{name}` needs an argument"))
    };

    let action = match name {
        "observe" => ComputerAction::Observe {
            include_screenshot: true,
        },
        "screenshot" => ComputerAction::Screenshot,
        "click" => ComputerAction::Click {
            x: num(0)?,
            y: num(1)?,
            button: MouseButton::Left,
        },
        "double_click" => ComputerAction::DoubleClick {
            x: num(0)?,
            y: num(1)?,
        },
        "right_click" => ComputerAction::RightClick {
            x: num(0)?,
            y: num(1)?,
        },
        "move" => ComputerAction::MoveCursor {
            x: num(0)?,
            y: num(1)?,
        },
        "scroll" => ComputerAction::Scroll {
            x: num(0)?,
            y: num(1)?,
            dx: num(2)?,
            dy: num(3)?,
        },
        "drag" => ComputerAction::Drag {
            from_x: num(0)?,
            from_y: num(1)?,
            to_x: num(2)?,
            to_y: num(3)?,
        },
        "type_text" => ComputerAction::TypeText {
            text: rest.join(" "),
        },
        "key_chord" => {
            if rest.is_empty() {
                return Err("`key_chord` needs at least one key".to_string());
            }
            ComputerAction::KeyChord {
                keys: rest.to_vec(),
            }
        }
        "ax_press" => ComputerAction::AxPress { element: first()? },
        "ax_set_value" => ComputerAction::AxSetValue {
            element: first()?,
            value: rest.get(1..).map(|v| v.join(" ")).unwrap_or_default(),
        },
        "wait" => ComputerAction::Wait {
            millis: first()?
                .parse::<u64>()
                .map_err(|e| format!("`wait`: bad milliseconds: {e}"))?,
        },
        "open_app" => {
            if rest.is_empty() {
                return Err("`open_app` needs an app name".to_string());
            }
            ComputerAction::OpenApp {
                name: rest.join(" "),
            }
        }
        other => return Err(format!("unknown action: `{other}`")),
    };
    Ok(action)
}
