use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::image::{Image, ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{AddressMode, AsBindGroup, AsBindGroupError, BindGroupLayout, BindGroupLayoutEntry, BindingResources, BindingType, BufferBindingType, BufferInitDescriptor, BufferUsages, FilterMode, OwnedBindingResource, SamplerBindingType, SamplerDescriptor, ShaderRef, ShaderStages, TextureSampleType, TextureViewDimension, UnpreparedBindGroup};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::{FallbackImage, GpuImage};
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;
use std::num::NonZeroU64;
use super::layers::TerrainMaterial;

pub const TERRAIN_SHADER_PATH: &str = "shaders/terrain_base.wgsl";
pub const SUPER_PERLIN_TEXTURE_PATH: &str = "textures/sperlin_rock.png";
pub const GRAINY_TEXTURE_PATH: &str = "textures/grainy.png";

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct TerrainBaseMaterialUniform {
    pub mode: u32,
    pub padding: [u32; 3], // Padding to ensure alignment
    pub time_x: f32,       // time
    pub time_y: f32,       // delta_time
    pub time_z: f32,       // sin(time)
    pub time_w: f32,       // cos(time)
}

impl Default for TerrainBaseMaterialUniform {
    fn default() -> Self {
        Self {
            mode: 0,
            padding: [0; 3],
            time_x: 0.0,
            time_y: 0.0,
            time_z: 0.0,
            time_w: 0.0,
        }
    }
}

#[derive(TypePath, Debug, Clone)]
pub(crate) struct TerrainBaseMaterial {
    pub material_uniform: TerrainBaseMaterialUniform,
    pub super_perlin_rock: Option<Handle<Image>>,
    pub grainy_texture: Option<Handle<Image>>,
}

impl TerrainMaterial for TerrainBaseMaterial {
    fn fragment_shader() -> ShaderRef { TERRAIN_SHADER_PATH.into() }
}

impl FromWorld for TerrainBaseMaterial {
    fn from_world(world: &mut World) -> Self {
        let image_repeat_settings = |s: &mut _| {
            *s = ImageLoaderSettings {
                sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    ..default()
                }),
                ..default()
            }
        };

        TerrainBaseMaterial {
            material_uniform: TerrainBaseMaterialUniform::default(),
            super_perlin_rock: Some(
                world
                    .load_asset_with_settings(SUPER_PERLIN_TEXTURE_PATH, image_repeat_settings),
            ),
            grainy_texture: Some(
                world.load_asset_with_settings(GRAINY_TEXTURE_PATH, image_repeat_settings),
            ),
        }
    }
}

impl AsBindGroup for TerrainBaseMaterial {
    type Data = ();
    type Param = (SRes<RenderAssets<GpuImage>>, SRes<FallbackImage>);

    fn label() -> Option<&'static str> {
        Some("terrain_material")
    }

    fn bindless_supported(_: &RenderDevice) -> bool { false }

    fn unprepared_bind_group(
        &self,
        _layout: &BindGroupLayout,
        render_device: &RenderDevice,
        (images, fallback_image): &mut SystemParamItem<'_, '_, Self::Param>,
        _: bool,
    ) -> Result<UnpreparedBindGroup<Self::Data>, AsBindGroupError> {
        let mut bindings = Vec::new();

        // Add uniform binding
        bindings.push((
            0,
            OwnedBindingResource::Buffer(render_device.create_buffer_with_data(
                &BufferInitDescriptor {
                    label: Some("terrain_material_uniform_buffer"),
                    contents: bytemuck::cast_slice(&[self.material_uniform]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                },
            )),
        ));

        // insert fallible texture-based entries at 0 so that if we fail here, we exit before allocating any buffers
        bindings.push((
            1,
            OwnedBindingResource::TextureView(
                TextureViewDimension::D2,
                {
                if let Some(texture_handle) = &self.super_perlin_rock {
                    images
                        .get(texture_handle)
                        .ok_or_else(|| AsBindGroupError::RetryNextUpdate)?
                        .texture_view
                        .clone()
                } else {
                    fallback_image.d2.texture_view.clone()
                }
            }),
        ));

        bindings.push((
            2,
            OwnedBindingResource::TextureView(
                TextureViewDimension::D2,
                {
                if let Some(texture_handle) = &self.grainy_texture {
                    images
                        .get(texture_handle)
                        .ok_or_else(|| AsBindGroupError::RetryNextUpdate)?
                        .texture_view
                        .clone()
                } else {
                    fallback_image.d2.texture_view.clone()
                }
            }),
        ));

        // 创建一个自定义的sampler，在UV方向上使用Repeat模式，并使用双线性插值
        let custom_sampler = render_device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat, // 虽然在2D中不使用，但设置它是个好习惯
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        // 使用自定义sampler而不是fallback_image的sampler
        bindings.push((3, OwnedBindingResource::Sampler(SamplerBindingType::Filtering, custom_sampler)));

        Ok(UnpreparedBindGroup { bindings: BindingResources(bindings), data: () })
    }

    fn bind_group_layout_entries(_render_device: &RenderDevice, _: bool) -> Vec<BindGroupLayoutEntry> {
        vec![
            // Uniform buffer
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        NonZeroU64::new(size_of::<TerrainBaseMaterialUniform>() as u64).unwrap(),
                    ),
                },
                count: None,
            },
            // Texture bindings
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Sampler bindings
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ]
    }
}
