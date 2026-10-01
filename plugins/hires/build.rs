fn main() {
    gst_plugin_version_helper::info();

    // cubecl → buildid references __mh_execute_header, which only exists in
    // executables. Allow it undefined when linking the GStreamer cdylib.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-U,__mh_execute_header");
    }
}
