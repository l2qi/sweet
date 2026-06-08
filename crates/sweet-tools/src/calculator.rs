// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_core::{ToolError, ToolFn};

/// Evaluate a simple arithmetic expression and return the result.
///
/// Supports `+`, `-`, `*`, `/`, `^` (power), parentheses, and decimal numbers.
#[derive(Default, serde::Deserialize, schemars::JsonSchema, sweet_core::Tool)]
#[tool(
    description = "Evaluate a simple arithmetic expression. Supports +, -, *, /, ^, parentheses, and decimal numbers.",
    risk = "readonly"
)]
pub struct Calculator {
    /// Arithmetic expression to evaluate, e.g. "(2 + 3) * 4".
    pub expression: String,
}

#[sweet_core::async_trait]
impl ToolFn for Calculator {
    async fn run(self) -> Result<String, ToolError> {
        let result = evaluate(&self.expression).map_err(|e| ToolError::Execution(e.into()))?;
        Ok(result.to_string())
    }
}

fn evaluate(input: &str) -> Result<f64, &'static str> {
    let mut parser = Parser::new(input);
    parser.expr().and_then(|v| {
        if parser.peek().is_none() {
            Ok(v)
        } else {
            Err("trailing characters")
        }
    })
}

struct Parser<'a> {
    chars: std::str::Chars<'a>,
    current: Option<char>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        let mut chars = input.chars();
        let current = chars.next();
        Self { chars, current }
    }

    fn advance(&mut self) {
        self.current = self.chars.next();
    }

    fn peek(&self) -> Option<char> {
        self.current
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        self.skip_whitespace();
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn number(&mut self) -> Result<f64, &'static str> {
        self.skip_whitespace();
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        if s.is_empty() || s == "." {
            return Err("expected number");
        }
        s.parse().map_err(|_| "invalid number")
    }

    fn factor(&mut self) -> Result<f64, &'static str> {
        self.skip_whitespace();
        if self.consume('(') {
            let v = self.expr()?;
            if !self.consume(')') {
                return Err("missing ')'");
            }
            return Ok(v);
        } else if let Some(c) = self.peek() {
            if c == '+' || c == '-' {
                let sign: f64 = if c == '+' { 1.0 } else { -1.0 };
                self.advance();
                return Ok(sign * self.factor()?);
            }
        }
        self.number()
    }

    fn power(&mut self) -> Result<f64, &'static str> {
        let mut left = self.factor()?;
        self.skip_whitespace();
        while self.consume('^') {
            let right = self.factor()?;
            left = left.powf(right);
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<f64, &'static str> {
        let mut left = self.power()?;
        self.skip_whitespace();
        loop {
            if self.consume('*') {
                left *= self.power()?;
            } else if self.consume('/') {
                let denom = self.power()?;
                if denom == 0.0 {
                    return Err("division by zero");
                }
                left /= denom;
            } else {
                break;
            }
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn expr(&mut self) -> Result<f64, &'static str> {
        let mut left = self.term()?;
        self.skip_whitespace();
        loop {
            if self.consume('+') {
                left += self.term()?;
            } else if self.consume('-') {
                left -= self.term()?;
            } else {
                break;
            }
            self.skip_whitespace();
        }
        Ok(left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::ToolSpec;

    #[test]
    fn name_and_description_are_present() {
        let t = ToolSpec::from(Calculator::default());
        assert_eq!(t.name, "calculator");
        assert!(!t.description.is_empty());
    }

    #[tokio::test]
    async fn basic_addition() {
        let tool = ToolSpec::from(Calculator::default());
        let result = tool
            .call(serde_json::json!({"expression": "2 + 3"}))
            .await
            .unwrap();
        assert_eq!(result, "5");
    }

    #[tokio::test]
    async fn mixed_expression() {
        let tool = ToolSpec::from(Calculator::default());
        let result = tool
            .call(serde_json::json!({"expression": "(2 + 3) * 4 - 6 / 2"}))
            .await
            .unwrap();
        assert_eq!(result, "17");
    }

    #[tokio::test]
    async fn power_and_decimal() {
        let tool = ToolSpec::from(Calculator::default());
        let result = tool
            .call(serde_json::json!({"expression": "2 ^ 3 + 1.5"}))
            .await
            .unwrap();
        assert_eq!(result, "9.5");
    }

    #[tokio::test]
    async fn division_by_zero() {
        let tool = ToolSpec::from(Calculator::default());
        let err = tool
            .call(serde_json::json!({"expression": "1 / 0"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("division by zero"));
    }

    #[tokio::test]
    async fn invalid_expression() {
        let tool = ToolSpec::from(Calculator::default());
        let err = tool
            .call(serde_json::json!({"expression": "2 +"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("expected number"));
    }
}
