//! GPU textures of immutable pixel arrays, keyed by the caller.
//!
//! Python gives every read-only pixel array a process-unique key
//! (`manim.utils.images.new_texture_key`) and takes a new one whenever it
//! replaces the array, so a key always names the same pixels.  The first draw
//! with a key uploads a texture; later draws reuse it.  Textures age out by
//! frame, as in WebRender's texture cache: one not drawn for `IDLE_FRAMES`
//! frames is dropped and is uploaded again if its key comes back.

use std::collections::HashMap;

use pyo3::PyResult;
use skia_safe::{gpu, Image};

/// Frames a texture may go undrawn before it is dropped.
const IDLE_FRAMES: u64 = 2;

struct Entry {
    image: Image,
    mipmapped: bool,
    last_used: u64,
}

#[derive(Default)]
pub struct TextureCache {
    frame: u64,
    entries: HashMap<u64, Entry>,
}

impl TextureCache {
    /// The texture for `key`, made from `raster()` on a miss or when mip
    /// levels are asked for and the cached texture has none.
    pub fn get(
        &mut self,
        context: &mut gpu::DirectContext,
        key: u64,
        mipmapped: bool,
        raster: impl FnOnce() -> PyResult<Image>,
    ) -> PyResult<Image> {
        if let Some(entry) = self.entries.get_mut(&key) {
            if entry.mipmapped || !mipmapped {
                entry.last_used = self.frame;
                return Ok(entry.image.clone());
            }
        }
        let raster = raster()?;
        let (raster, levels) = if mipmapped {
            (
                raster.with_default_mipmaps().unwrap_or(raster),
                gpu::Mipmapped::Yes,
            )
        } else {
            (raster, gpu::Mipmapped::No)
        };
        // A texture-backed image frees the host copy and stays resident; the
        // raster image is the fallback when the upload is refused.
        let image = raster.new_texture_image(context, levels).unwrap_or(raster);
        self.entries.insert(
            key,
            Entry {
                image: image.clone(),
                mipmapped,
                last_used: self.frame,
            },
        );
        Ok(image)
    }

    /// Close a frame: drop the textures that have gone undrawn too long.
    pub fn end_frame(&mut self) {
        let frame = self.frame;
        self.entries
            .retain(|_, entry| frame - entry.last_used < IDLE_FRAMES);
        self.frame += 1;
    }
}
