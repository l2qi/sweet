// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Raw bindings to the macOS C frameworks, plus small safe helpers.
//!
//! Every type here is an opaque pointer or a `#[repr(C)]` struct matching the
//! platform ABI on 64-bit macOS (`CGFloat` = `f64`, `CFIndex`/`CFTypeID` =
//! `long`/`unsigned long`). The frameworks are linked by `build.rs`.
//!
//! CoreFoundation ownership rule: functions named `Create`/`Copy` return a
//! retained (+1) reference the caller must `CFRelease`; [`CfBox`] owns such a
//! reference and releases it on drop. `Get` functions return *borrowed*
//! references that must NOT be released and are only valid while their owner
//! lives.

#![allow(non_snake_case, non_upper_case_globals)]

use std::os::raw::{c_char, c_long, c_ulong, c_void};

// ---- Opaque CoreFoundation / Quartz / Accessibility pointer types ----
//
// All are `const struct __Foo*` in C. We alias them to a single raw pointer
// type; the layer's functions keep usage disciplined.
pub type CFTypeRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CFArrayRef = *const c_void;
pub type CFDictionaryRef = *const c_void;
pub type CFNumberRef = *const c_void;
pub type CFBooleanRef = *const c_void;
pub type CFAllocatorRef = *const c_void;
pub type CFMutableDataRef = *const c_void;
pub type CFDataRef = *const c_void;

pub type CGImageRef = *const c_void;
pub type CGEventRef = *const c_void;
pub type CGEventSourceRef = *const c_void;
pub type CGImageDestinationRef = *const c_void;

pub type AXUIElementRef = *const c_void;
pub type AXValueRef = *const c_void;

// ---- Scalar ABI aliases ----
pub type CFIndex = c_long;
pub type CFTypeID = c_ulong;
/// CoreFoundation `Boolean` is `unsigned char`.
pub type Boolean = u8;
pub type CFStringEncoding = u32;
pub type CGFloat = f64;
pub type CGDirectDisplayID = u32;
pub type CGWindowID = u32;
pub type CGWindowListOption = u32;
pub type CGKeyCode = u16;
pub type CGError = i32;
pub type AXError = i32;
pub type AXValueType = u32;
/// `pid_t`.
pub type Pid = i32;

// ---- #[repr(C)] geometry ----
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CGPoint {
    pub x: CGFloat,
    pub y: CGFloat,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CGSize {
    pub width: CGFloat,
    pub height: CGFloat,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CGRect {
    pub origin: CGPoint,
    pub size: CGSize,
}

// ---- Constants ----
pub const kCFStringEncodingUTF8: CFStringEncoding = 0x0800_0100;
/// `kCFNumberSInt64Type`. `CFNumberType` is `CF_ENUM(CFIndex, ...)`, i.e. a
/// `long` (8 bytes on 64-bit), not an `int` - passing a 32-bit value would
/// leave the high bits unspecified on arm64 and the tag would not match.
pub const kCFNumberSInt64Type: CFIndex = 4;

pub const kCGHIDEventTap: u32 = 0;

// CGEventType
pub const kCGEventLeftMouseDown: u32 = 1;
pub const kCGEventLeftMouseUp: u32 = 2;
pub const kCGEventRightMouseDown: u32 = 3;
pub const kCGEventRightMouseUp: u32 = 4;
pub const kCGEventMouseMoved: u32 = 5;
pub const kCGEventLeftMouseDragged: u32 = 6;
pub const kCGEventOtherMouseDown: u32 = 25;
pub const kCGEventOtherMouseUp: u32 = 26;

// CGMouseButton
pub const kCGMouseButtonLeft: u32 = 0;
pub const kCGMouseButtonRight: u32 = 1;
pub const kCGMouseButtonCenter: u32 = 2;

// CGEventFlags
pub const kCGEventFlagMaskShift: u64 = 0x0002_0000;
pub const kCGEventFlagMaskControl: u64 = 0x0004_0000;
pub const kCGEventFlagMaskAlternate: u64 = 0x0008_0000;
pub const kCGEventFlagMaskCommand: u64 = 0x0010_0000;
pub const kCGEventFlagMaskSecondaryFn: u64 = 0x0080_0000;

// CGScrollEventUnit
pub const kCGScrollEventUnitLine: u32 = 1;
// CGEventField
pub const kCGMouseEventClickState: u32 = 1;

// CGWindowListOption / CGWindowID
pub const kCGWindowListOptionOnScreenOnly: u32 = 1 << 0;
pub const kCGWindowListExcludeDesktopElements: u32 = 1 << 4;
pub const kCGNullWindowID: u32 = 0;

// AXValueType
pub const kAXValueCGPointType: AXValueType = 1;
pub const kAXValueCGSizeType: AXValueType = 2;
// AXError
pub const kAXErrorSuccess: AXError = 0;

extern "C" {
    // ---- CoreFoundation ----
    pub fn CFRelease(cf: CFTypeRef);
    pub fn CFGetTypeID(cf: CFTypeRef) -> CFTypeID;

    pub fn CFStringCreateWithBytes(
        alloc: CFAllocatorRef,
        bytes: *const u8,
        num_bytes: CFIndex,
        encoding: CFStringEncoding,
        is_external_representation: Boolean,
    ) -> CFStringRef;
    pub fn CFStringGetLength(s: CFStringRef) -> CFIndex;
    pub fn CFStringGetCString(
        s: CFStringRef,
        buffer: *mut c_char,
        buffer_size: CFIndex,
        encoding: CFStringEncoding,
    ) -> Boolean;
    pub fn CFStringGetTypeID() -> CFTypeID;

    pub fn CFArrayGetCount(arr: CFArrayRef) -> CFIndex;
    pub fn CFArrayGetValueAtIndex(arr: CFArrayRef, idx: CFIndex) -> *const c_void;
    pub fn CFArrayGetTypeID() -> CFTypeID;

    pub fn CFBooleanGetValue(b: CFBooleanRef) -> Boolean;
    pub fn CFBooleanGetTypeID() -> CFTypeID;

    pub fn CFNumberGetValue(n: CFNumberRef, the_type: CFIndex, value_ptr: *mut c_void) -> Boolean;
    pub fn CFNumberGetTypeID() -> CFTypeID;

    pub fn CFDictionaryGetValue(d: CFDictionaryRef, key: *const c_void) -> *const c_void;

    pub fn CFDataCreateMutable(alloc: CFAllocatorRef, capacity: CFIndex) -> CFMutableDataRef;
    pub fn CFDataGetLength(d: CFDataRef) -> CFIndex;
    pub fn CFDataGetBytePtr(d: CFDataRef) -> *const u8;

    // ---- CoreGraphics: Quartz events ----
    pub fn CGEventCreateMouseEvent(
        source: CGEventSourceRef,
        mouse_type: u32,
        mouse_cursor_position: CGPoint,
        mouse_button: u32,
    ) -> CGEventRef;
    pub fn CGEventCreateKeyboardEvent(
        source: CGEventSourceRef,
        virtual_key: CGKeyCode,
        key_down: bool,
    ) -> CGEventRef;
    pub fn CGEventKeyboardSetUnicodeString(
        event: CGEventRef,
        string_length: c_ulong,
        unicode_string: *const u16,
    );
    pub fn CGEventCreateScrollWheelEvent(
        source: CGEventSourceRef,
        units: u32,
        wheel_count: u32,
        wheel1: i32,
        ...
    ) -> CGEventRef;
    pub fn CGEventSetFlags(event: CGEventRef, flags: u64);
    pub fn CGEventSetIntegerValueField(event: CGEventRef, field: u32, value: i64);
    pub fn CGEventPost(tap: u32, event: CGEventRef);
    pub fn CGWarpMouseCursorPosition(new_cursor_position: CGPoint) -> CGError;
    pub fn CGEventCreate(source: CGEventSourceRef) -> CGEventRef;
    pub fn CGEventGetLocation(event: CGEventRef) -> CGPoint;

    // ---- CoreGraphics: display, capture, window list ----
    pub fn CGMainDisplayID() -> CGDirectDisplayID;
    pub fn CGDisplayBounds(display: CGDirectDisplayID) -> CGRect;
    pub fn CGDisplayCreateImage(display: CGDirectDisplayID) -> CGImageRef;
    pub fn CGImageGetWidth(image: CGImageRef) -> usize;
    pub fn CGImageGetHeight(image: CGImageRef) -> usize;
    pub fn CGPreflightScreenCaptureAccess() -> bool;
    pub fn CGRequestScreenCaptureAccess() -> bool;
    pub fn CGWindowListCopyWindowInfo(
        option: CGWindowListOption,
        relative_to_window: CGWindowID,
    ) -> CFArrayRef;
    pub fn CGRectMakeWithDictionaryRepresentation(dict: CFDictionaryRef, rect: *mut CGRect)
        -> bool;

    // ---- ImageIO ----
    pub fn CGImageDestinationCreateWithData(
        data: CFMutableDataRef,
        type_id: CFStringRef,
        count: usize,
        options: CFDictionaryRef,
    ) -> CGImageDestinationRef;
    pub fn CGImageDestinationAddImage(
        idst: CGImageDestinationRef,
        image: CGImageRef,
        properties: CFDictionaryRef,
    );
    pub fn CGImageDestinationFinalize(idst: CGImageDestinationRef) -> bool;

    // ---- ApplicationServices: Accessibility ----
    pub fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    pub fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> AXError;
    pub fn AXUIElementCopyActionNames(element: AXUIElementRef, names: *mut CFArrayRef) -> AXError;
    pub fn AXUIElementPerformAction(element: AXUIElementRef, action: CFStringRef) -> AXError;
    pub fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> AXError;
    pub fn AXUIElementGetPid(element: AXUIElementRef, pid: *mut Pid) -> AXError;
    pub fn AXValueGetValue(
        value: AXValueRef,
        the_type: AXValueType,
        value_ptr: *mut c_void,
    ) -> Boolean;
    pub fn AXUIElementGetTypeID() -> CFTypeID;
    pub fn AXValueGetTypeID() -> CFTypeID;
    pub fn AXIsProcessTrusted() -> Boolean;
}

// ---------------------------------------------------------------------------
// Safe helpers
// ---------------------------------------------------------------------------

/// Owns a retained CoreFoundation reference and releases it on drop.
pub struct CfBox(CFTypeRef);

impl CfBox {
    /// Wrap a +1 reference (e.g. from a `Create`/`Copy` call). Returns `None`
    /// for a null pointer.
    pub fn from_create(p: CFTypeRef) -> Option<CfBox> {
        if p.is_null() {
            None
        } else {
            Some(CfBox(p))
        }
    }

    /// The borrowed pointer. Valid only while this `CfBox` lives.
    pub fn as_ptr(&self) -> CFTypeRef {
        self.0
    }
}

impl Drop for CfBox {
    fn drop(&mut self) {
        // Safety: `self.0` is a non-null retained reference (invariant of
        // `from_create`) that we own exactly once.
        unsafe { CFRelease(self.0) }
    }
}

/// Create an owned `CFString` from a Rust string slice.
///
/// Panics only on the impossible case of CoreFoundation rejecting valid UTF-8
/// bytes - every call site passes a literal attribute/key name or model-given
/// text, so a null here would be an unrecoverable platform invariant break.
pub fn cfstr(s: &str) -> CfBox {
    // Safety: bytes/length describe a valid UTF-8 buffer; allocator is default.
    let p = unsafe {
        CFStringCreateWithBytes(
            std::ptr::null(),
            s.as_ptr(),
            s.len() as CFIndex,
            kCFStringEncodingUTF8,
            0,
        )
    };
    CfBox::from_create(p).expect("CFStringCreateWithBytes returned null for valid UTF-8")
}

/// Copy a `CFString`'s contents into a Rust `String`.
pub fn cf_string_to_string(s: CFStringRef) -> Option<String> {
    if s.is_null() {
        return None;
    }
    // Safety: `s` is a valid CFStringRef.
    let len = unsafe { CFStringGetLength(s) };
    if len < 0 {
        return None;
    }
    // UTF-16 length -> UTF-8 needs at most 4 bytes per unit, +1 for the NUL.
    let cap = (len as usize).saturating_mul(4).saturating_add(1);
    let mut buf = vec![0u8; cap];
    // Safety: buffer is `cap` bytes; CFString fills up to `cap` and NUL-terminates.
    let ok = unsafe {
        CFStringGetCString(
            s,
            buf.as_mut_ptr() as *mut c_char,
            cap as CFIndex,
            kCFStringEncodingUTF8,
        )
    };
    if ok == 0 {
        return None;
    }
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(cap);
    buf.truncate(nul);
    String::from_utf8(buf).ok()
}

#[inline]
fn type_id(cf: CFTypeRef) -> CFTypeID {
    // Safety: `cf` is a valid CF reference.
    unsafe { CFGetTypeID(cf) }
}

pub fn is_string(cf: CFTypeRef) -> bool {
    !cf.is_null() && type_id(cf) == unsafe { CFStringGetTypeID() }
}
pub fn is_array(cf: CFTypeRef) -> bool {
    !cf.is_null() && type_id(cf) == unsafe { CFArrayGetTypeID() }
}
pub fn is_bool(cf: CFTypeRef) -> bool {
    !cf.is_null() && type_id(cf) == unsafe { CFBooleanGetTypeID() }
}
pub fn is_number(cf: CFTypeRef) -> bool {
    !cf.is_null() && type_id(cf) == unsafe { CFNumberGetTypeID() }
}
pub fn is_ax_element(cf: CFTypeRef) -> bool {
    !cf.is_null() && type_id(cf) == unsafe { AXUIElementGetTypeID() }
}
pub fn is_ax_value(cf: CFTypeRef) -> bool {
    !cf.is_null() && type_id(cf) == unsafe { AXValueGetTypeID() }
}

/// Copy an accessibility attribute, returning the owned value (or `None` if the
/// attribute is unsupported / empty).
pub fn copy_attr(el: AXUIElementRef, attr: &str) -> Option<CfBox> {
    let key = cfstr(attr);
    let mut value: CFTypeRef = std::ptr::null();
    // Safety: `el` and the attribute string are valid; `value` is an out-param.
    let err = unsafe { AXUIElementCopyAttributeValue(el, key.as_ptr(), &mut value) };
    if err == kAXErrorSuccess && !value.is_null() {
        CfBox::from_create(value)
    } else {
        None
    }
}

/// Read a `CFBoolean` value.
pub fn cf_bool(cf: CFTypeRef) -> Option<bool> {
    if is_bool(cf) {
        // Safety: confirmed CFBoolean.
        Some(unsafe { CFBooleanGetValue(cf) } != 0)
    } else {
        None
    }
}

/// Read a `CFNumber` as `i64`.
pub fn cf_i64(cf: CFTypeRef) -> Option<i64> {
    if !is_number(cf) {
        return None;
    }
    let mut n: i64 = 0;
    // Safety: confirmed CFNumber; reading into a 64-bit slot as SInt64.
    let ok =
        unsafe { CFNumberGetValue(cf, kCFNumberSInt64Type, &mut n as *mut i64 as *mut c_void) };
    if ok != 0 {
        Some(n)
    } else {
        None
    }
}

/// Extract a `CGPoint` from an `AXValue`.
pub fn ax_point(cf: AXValueRef) -> Option<CGPoint> {
    if !is_ax_value(cf) {
        return None;
    }
    let mut p = CGPoint::default();
    // Safety: confirmed AXValue; type tag matches the out slot.
    let ok = unsafe {
        AXValueGetValue(
            cf,
            kAXValueCGPointType,
            &mut p as *mut CGPoint as *mut c_void,
        )
    };
    if ok != 0 {
        Some(p)
    } else {
        None
    }
}

/// Extract a `CGSize` from an `AXValue`.
pub fn ax_size(cf: AXValueRef) -> Option<CGSize> {
    if !is_ax_value(cf) {
        return None;
    }
    let mut s = CGSize::default();
    // Safety: confirmed AXValue; type tag matches the out slot.
    let ok =
        unsafe { AXValueGetValue(cf, kAXValueCGSizeType, &mut s as *mut CGSize as *mut c_void) };
    if ok != 0 {
        Some(s)
    } else {
        None
    }
}
