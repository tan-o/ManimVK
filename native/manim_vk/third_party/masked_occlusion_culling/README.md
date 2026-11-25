# Masked Occlusion Culling

Intel's library from https://github.com/GameTechDev/MaskedOcclusionCulling
(Apache-2.0, `license.txt`): `MaskedOcclusionCulling.{h,cpp}`,
`MaskedOcclusionCullingAVX2.cpp`, `MaskedOcclusionCullingAVX512.cpp`,
`MaskedOcclusionCullingCommon.inl` and `CompilerSpecific.inl`, unchanged
except that `CompilerSpecific.inl` maps `__cpuidex` and `_xgetbv` to its own
functions on GCC and Clang, which declare those names themselves.
`moc_capi.cpp` is ManimVK's C interface to it.
