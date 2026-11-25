//! Builds Intel's Masked Occlusion Culling (see `src/occlusion.rs`) on x86,
//! each file with the instruction set it is written for, as its CMake build
//! does; the library picks the best one at run time.

fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "x86_64" && arch != "x86" {
        return;
    }
    let dir = std::path::Path::new("third_party/masked_occlusion_culling");
    println!("cargo:rerun-if-changed={}", dir.display());
    let files: [(&str, &[&str], &[&str]); 4] = [
        ("moc_capi.cpp", &["-msse4.1"], &[]),
        ("MaskedOcclusionCulling.cpp", &["-msse4.1"], &[]),
        ("MaskedOcclusionCullingAVX2.cpp", &["-mavx2", "-mfma", "-msse4.1"], &["/arch:AVX2"]),
        ("MaskedOcclusionCullingAVX512.cpp", &["-mavx2", "-mfma", "-msse4.1"], &["/arch:AVX2"]),
    ];
    for (file, gnu, msvc) in files {
        let mut build = cc::Build::new();
        build.cpp(true).std("c++17").include(dir).file(dir.join(file)).warnings(false);
        let flags = if build.get_compiler().is_like_msvc() { msvc } else { gnu };
        for flag in flags {
            build.flag(flag);
        }
        build.compile(&format!("moc_{}", file.trim_end_matches(".cpp").to_lowercase()));
    }
}
