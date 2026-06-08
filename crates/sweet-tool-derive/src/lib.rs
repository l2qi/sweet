// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{parse_macro_input, spanned::Spanned, DeriveInput, Expr, Lit, Meta, MetaNameValue};

#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;

    // Parse #[tool(...)] attributes
    let mut name_override = None;
    let mut description = None;
    let mut risk = None;
    for attr in &input.attrs {
        if !attr.path().is_ident("tool") {
            continue;
        }
        let nested = match &attr.meta {
            Meta::List(l) => &l.tokens,
            _ => {
                return syn::Error::new(attr.span(), "expected #[tool(...)]")
                    .to_compile_error()
                    .into();
            }
        };
        let parser = syn::punctuated::Punctuated::<MetaNameValue, syn::Token![,]>::parse_terminated;
        let kvs = match parser.parse2(nested.clone()) {
            Ok(kvs) => kvs,
            Err(e) => return e.to_compile_error().into(),
        };
        for nv in kvs {
            if nv.path.is_ident("name") {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(ref s) = expr_lit.lit {
                        name_override = Some(s.value());
                    }
                }
            } else if nv.path.is_ident("description") {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(ref s) = expr_lit.lit {
                        description = Some(s.value());
                    }
                }
            } else if nv.path.is_ident("risk") {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(ref s) = expr_lit.lit {
                        risk = Some(s.value());
                    }
                }
            }
        }
    }

    let description = match description {
        Some(d) => d,
        None => {
            return syn::Error::new(
                input.span(),
                "#[derive(Tool)] requires #[tool(description = \"...\")]",
            )
            .to_compile_error()
            .into();
        }
    };

    let tool_name = match name_override {
        Some(n) => n,
        None => snake_case(&struct_name.to_string()),
    };

    let risk_expr = match risk.as_deref() {
        Some("readonly") => quote! { ::sweet_core::ToolRisk::ReadOnly },
        Some("file_write") => quote! { ::sweet_core::ToolRisk::FileWrite },
        Some("dangerous") => quote! { ::sweet_core::ToolRisk::Dangerous },
        Some(other) => {
            return syn::Error::new(
                input.span(),
                format!(
                    "unknown risk level `{other}`; expected \"readonly\", \"file_write\", or \"dangerous\""
                ),
            )
            .to_compile_error()
            .into();
        }
        None => quote! { ::sweet_core::ToolRisk::Dangerous },
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        #[::sweet_core::__private::async_trait]
        impl #impl_generics ::sweet_core::ToolHandler for #struct_name #ty_generics #where_clause {
            async fn call(
                &self,
                args: ::sweet_core::__private::serde_json::Value,
            ) -> ::std::result::Result<::std::string::String, ::sweet_core::ToolError> {
                let parsed: #struct_name = ::sweet_core::__private::serde_json::from_value(args)
                    .map_err(::sweet_core::ToolError::InvalidArgs)?;
                let result = <#struct_name as ::sweet_core::ToolFn>::run(parsed).await?;
                ::std::result::Result::Ok(result)
            }
        }

        impl #impl_generics ::std::convert::From<#struct_name #ty_generics> for ::sweet_core::ToolSpec #where_clause {
            fn from(tool: #struct_name #ty_generics) -> Self {
                let schema = ::sweet_core::__private::schema_for!(#struct_name);
                ::sweet_core::ToolSpec::new(
                    #tool_name,
                    #description,
                    ::sweet_core::__private::serde_json::to_value(schema)
                        .expect("derive(Tool): schemars produced a non-Value-serializable schema"),
                    tool,
                )
                .with_risk(#risk_expr)
            }
        }
    };

    TokenStream::from(expanded)
}

fn snake_case(s: &str) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}
