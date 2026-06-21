// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Main-display screen capture, PNG-encoded via ImageIO, resized to the
//! display's logical point size, and saved.
//!
//! Uses `CGDisplayCreateImage` (CoreGraphics). It is deprecated in macOS 14 in
//! favor of ScreenCaptureKit, but remains functional and is far simpler to bind
//! than the async ScreenCaptureKit API; capture is isolated here so a future
//! migration is local to this file.
//!
//! `CGDisplayCreateImage` captures in **physical pixels**, so on a HiDPI/Retina
//! display the image is larger than the **logical point** size that
//! `CGDisplayBounds`, the AX element frames, and `CGEvent` clicks all use - and
//! by a factor that depends on the user's display *and* scaling mode (a 14" MBP
//! set to "More Space" reports 1800x1169 points but captures at ~2x that). We
//! resize the capture down to the point size read live from `CGDisplayBounds`
//! (no hardcoded resolution), so a position the model reads off the screenshot
//! maps 1:1 to a click point. Without this, clicks are off by the scale factor.
//! It also shrinks what would otherwise be a multi-megabyte full-res PNG.

use std::path::Path;

use sweet_computer_use_core::{crosshair_rects, ComputerUseError, Point, Screenshot, Size};

use super::ffi::{self, CGImageRef, CfBox};
use super::permissions;

/// Capture the main display, save a PNG under `dir`, and return the bytes +
/// path. `screen_size` is the display's logical point size (from
/// `CGDisplayBounds`); the physical-pixel capture is resized to it so the
/// screenshot shares the click/AX coordinate space. When `cursor` is given, a
/// crosshair marking it is drawn on the screenshot so the model can aim relative
/// to where the pointer currently is. Fails with
/// [`ComputerUseError::PermissionDenied`] if Screen Recording is not granted.
pub fn capture_main_display(
    dir: &Path,
    screen_size: Size,
    cursor: Option<Point>,
) -> Result<Screenshot, ComputerUseError> {
    if !permissions::screen_capture_allowed() {
        // Trigger the prompt so the grant is available after a restart.
        permissions::request_screen_capture();
        return Err(ComputerUseError::PermissionDenied(
            permissions::screen_capture_hint(),
        ));
    }

    let display = unsafe { ffi::CGMainDisplayID() };
    let image =
        CfBox::from_create(unsafe { ffi::CGDisplayCreateImage(display) }).ok_or_else(|| {
            ComputerUseError::Platform(
                "screen capture returned no image (CGDisplayCreateImage). On macOS 14+ this can \
                 happen even with Screen Recording granted; a ScreenCaptureKit-based capture path \
                 may be required."
                    .into(),
            )
        })?;

    let capture_w = unsafe { ffi::CGImageGetWidth(image.as_ptr()) } as u32;
    let capture_h = unsafe { ffi::CGImageGetHeight(image.as_ptr()) } as u32;
    let raw_png = encode_png(image.as_ptr())?;

    // Resize the physical-pixel capture to the display's logical point size (so
    // the screenshot shares the click coordinate space) and draw the cursor
    // marker. `screenshot_target` is `None` when no resize is needed.
    let target = screen_size.screenshot_target(capture_w, capture_h);
    let (png, width, height) = render(raw_png, capture_w, capture_h, target, cursor);
    let path = save(dir, &png)?;

    Ok(Screenshot {
        data: png,
        media_type: "image/png".to_string(),
        path: Some(path),
        width,
        height,
    })
}

/// Resize the capture to `target` (if any) and draw the cursor crosshair (if
/// any), returning the PNG bytes and final dimensions. Any decode/encode failure
/// falls back to the original capture bytes and `capture_*` dimensions, so a
/// processing hiccup never drops the screenshot. With nothing to do, the raw PNG
/// is returned untouched.
fn render(
    raw_png: Vec<u8>,
    capture_w: u32,
    capture_h: u32,
    target: Option<(u32, u32)>,
    cursor: Option<Point>,
) -> (Vec<u8>, u32, u32) {
    if target.is_none() && cursor.is_none() {
        return (raw_png, capture_w, capture_h);
    }
    let decoded = match image::load_from_memory(&raw_png) {
        Ok(img) => img,
        Err(_) => return (raw_png, capture_w, capture_h),
    };
    let resized = match target {
        Some((tw, th)) => decoded.resize(tw, th, image::imageops::FilterType::Triangle),
        None => decoded,
    };
    let mut rgba = resized.into_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    if let Some(c) = cursor {
        // The image is point-sized, so cursor point coordinates are image pixels.
        draw_crosshair(&mut rgba, c.x, c.y);
    }
    let mut out = Vec::new();
    let dynimg = image::DynamicImage::ImageRgba8(rgba);
    match dynimg.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png) {
        Ok(()) => (out, w, h),
        Err(_) => (raw_png, capture_w, capture_h),
    }
}

/// Draw a magenta, black-outlined crosshair at `(cx, cy)` so the model can see
/// where the cursor points and correct relative to it. The outline keeps it
/// visible on any background.
fn draw_crosshair(img: &mut image::RgbaImage, cx: f64, cy: f64) {
    const HALO: [u8; 4] = [0, 0, 0, 255];
    const CORE: [u8; 4] = [255, 0, 255, 255];
    let rects = crosshair_rects(cx, cy, img.width(), img.height());
    // Outline first (each rect grown by 1px), then the colored core on top.
    for r in &rects {
        fill_rect(
            img,
            r.x as i64 - 1,
            r.y as i64 - 1,
            r.width as i64 + 2,
            r.height as i64 + 2,
            HALO,
        );
    }
    for r in &rects {
        fill_rect(
            img,
            r.x as i64,
            r.y as i64,
            r.width as i64,
            r.height as i64,
            CORE,
        );
    }
}

/// Fill the rectangle `(x, y, w, h)` with `color`, clipped to the image bounds.
fn fill_rect(img: &mut image::RgbaImage, x: i64, y: i64, w: i64, h: i64, color: [u8; 4]) {
    let iw = img.width() as i64;
    let ih = img.height() as i64;
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + w).min(iw);
    let y1 = (y + h).min(ih);
    for yy in y0..y1 {
        for xx in x0..x1 {
            img.put_pixel(xx as u32, yy as u32, image::Rgba(color));
        }
    }
}

/// Encode a `CGImage` to PNG bytes via an ImageIO destination backed by a
/// `CFMutableData`.
fn encode_png(image: CGImageRef) -> Result<Vec<u8>, ComputerUseError> {
    let data = CfBox::from_create(unsafe { ffi::CFDataCreateMutable(std::ptr::null(), 0) })
        .ok_or_else(|| ComputerUseError::Platform("CFDataCreateMutable failed".into()))?;
    let png_type = ffi::cfstr("public.png").ok_or_else(|| {
        ComputerUseError::Platform("CFStringCreateWithBytes failed for public.png".into())
    })?;

    let dest = CfBox::from_create(unsafe {
        ffi::CGImageDestinationCreateWithData(data.as_ptr(), png_type.as_ptr(), 1, std::ptr::null())
    })
    .ok_or_else(|| ComputerUseError::Platform("CGImageDestinationCreateWithData failed".into()))?;

    unsafe { ffi::CGImageDestinationAddImage(dest.as_ptr(), image, std::ptr::null()) };
    if !unsafe { ffi::CGImageDestinationFinalize(dest.as_ptr()) } {
        return Err(ComputerUseError::Platform(
            "CGImageDestinationFinalize failed".into(),
        ));
    }

    let len = unsafe { ffi::CFDataGetLength(data.as_ptr()) };
    let ptr = unsafe { ffi::CFDataGetBytePtr(data.as_ptr()) };
    if ptr.is_null() || len <= 0 {
        return Err(ComputerUseError::Platform(
            "PNG encoding produced no data".into(),
        ));
    }
    // Safety: `ptr` points to `len` valid bytes owned by `data`, which is alive.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
    Ok(bytes)
}

fn save(dir: &Path, png: &[u8]) -> Result<String, ComputerUseError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| ComputerUseError::Platform(format!("create screenshot dir: {e}")))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("screenshot-{nanos}.png"));
    std::fs::write(&path, png)
        .map_err(|e| ComputerUseError::Platform(format!("write screenshot: {e}")))?;
    Ok(path.to_string_lossy().into_owned())
}
