// Link the macOS system frameworks the FFI layer binds against. This runs on
// the build host; on non-macOS targets it emits nothing and the crate compiles
// as an `Unsupported` stub (see `provider.rs`).
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        // CoreFoundation: CFString/CFArray/CFDictionary/CFNumber/CFData.
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        // CoreGraphics: Quartz events, display capture, window list.
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        // ApplicationServices: the Accessibility (AXUIElement) API.
        println!("cargo:rustc-link-lib=framework=ApplicationServices");
        // ImageIO: CGImageDestination PNG encoding.
        println!("cargo:rustc-link-lib=framework=ImageIO");
    }
}
