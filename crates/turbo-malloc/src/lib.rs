use std::{
    alloc::{GlobalAlloc, Layout},
    ptr,
};

/// Turbo's preferred global allocator. This is a new type instead of a type
/// alias because you can't use type aliases to instantiate unit types (E0423).
pub struct TurboMalloc;

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
impl TurboMalloc {
    pub fn collect() {
        unsafe {
            libmimalloc_sys::mi_collect(false);
        }
    }

    pub fn print_stats() {
        unsafe {
            libmimalloc_sys::mi_stats_print(ptr::null_mut());
        }
    }

    pub fn memory_usage() -> usize {
        let mut current_commit: usize = 0;
        unsafe {
            libmimalloc_sys::mi_process_info(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut current_commit as *mut _,
                ptr::null_mut(),
                ptr::null_mut(),
            );
        }
        current_commit
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
unsafe impl GlobalAlloc for TurboMalloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        mimalloc::MiMalloc.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        mimalloc::MiMalloc.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        mimalloc::MiMalloc.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        mimalloc::MiMalloc.realloc(ptr, layout, new_size)
    }
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe impl GlobalAlloc for TurboMalloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        std::alloc::System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        std::alloc::System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        std::alloc::System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        std::alloc::System.realloc(ptr, layout, new_size)
    }
}
