use bevy::prelude::*;
use bevy::render::{
    render_resource::{AsBindGroup,ShaderRef},
    view::RenderLayers
};

pub trait TerrainMaterial: AsBindGroup + Clone + Sized + FromWorld + Send + Sync {
    fn fragment_shader() -> ShaderRef { ShaderRef::Default }
}