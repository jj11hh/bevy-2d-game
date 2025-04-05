use bevy::asset::Assets;
use bevy::core_pipeline::core_2d::Transparent2d;
use bevy::ecs::entity::EntityHash;
use bevy::ecs::system::{StaticSystemParam, SystemState};
use bevy::image::{
    ImageSampler, TextureFormatPixelInfo,
};
use bevy::math::{vec2, vec3, FloatOrd};
use bevy::pbr::MeshFlags;
use bevy::prelude::*;
use bevy::render::batching::NoAutomaticBatching;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::mesh::{MeshVertexBufferLayoutRef, RenderMesh};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItemExtraIndex, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d};
use bevy::render::render_resource::{
    AddressMode, AsBindGroup, BindGroupEntries, BindGroupLayoutEntries, DefaultImageSampler, Extent3d, FilterMode, ImageDataLayout, IntoBinding,
    OwnedBindingResource, PipelineCache, RenderPipelineDescriptor,
    SamplerBindingType, SamplerDescriptor, ShaderRef, ShaderStages, SpecializedMeshPipeline,
    SpecializedMeshPipelineError, SpecializedMeshPipelines, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::sync_component::SyncComponentPlugin;
use bevy::render::sync_world::{MainEntity, MainEntityHashSet, RenderEntity};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, RenderVisibleEntities};
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::render::sync_world::MainEntityHashMap;
use bevy::sprite::{
    extract_mesh2d, Material2dBindGroupId, Mesh2dPipeline, Mesh2dPipelineKey,
    Mesh2dTransforms, RenderMesh2dInstance,
};
use bevy::tasks::{ComputeTaskPool, ParallelSliceMut};
use bevy::utils::HashMap;
use bytemuck::{Pod, Zeroable};
use noise_functions::modifiers::{Fbm, Frequency};
use noise_functions::{Constant, Noise, OpenSimplex2, Sample};
use std::mem::size_of;
use strum::FromRepr;

mod layers;
mod base_material;
mod draw;

use self::layers::TerrainMaterial;
use self::base_material::TerrainBaseMaterial;
use self::draw::*;

pub type DrawTerrainBaseMesh2d = DrawTerrainMesh2d<TerrainBaseMaterial>;

const RANDOM_SEED: u32 = 1;
const SEA_LEVEL: f32 = 64.0f32;
const HEIGHT_LIMIT: f32 = 255.0f32;

#[derive(Resource, Deref, DerefMut, Default)]
pub struct TerrainPerChunkDataStore(pub MainEntityHashMap<TerrainPerChunkData>);

#[derive(Resource, Deref, DerefMut, Default)]
pub struct TerrainRendererVersions(pub MainEntityHashMap<usize>);

#[derive(Resource)]
pub struct NoiseSource<T> {
    noise: T,
}

const ISLAND_CHUNK_SIZE: u32 = 128;

const ISLAND_RADIUS: f32 = 256.0;

pub const TILE_PIXEL_SIZE: f32 = 32.0;

const MAX_RENDER_MODE: u32 = 2;

#[derive(Resource, ExtractResource, Clone, Copy, Default)]
pub struct TerrainRenderMode {
    pub mode: u32,
}


pub fn extract_terrain_chunk<TrunkType: AsTextureProvider + Component>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut data_store: ResMut<TerrainPerChunkDataStore>,
    query: Extract<
        Query<(
            Entity,
            &RenderEntity,
            &TrunkType,
            Option<&TerrainChunkChanged>,
        )>,
    >,
) {
    trace!("Extracting terrain chunks");

    // Collect active entities to clean up data for destroyed entities later
    let mut active_entities = MainEntityHashSet::with_capacity_and_hasher(0, EntityHash::default());

    for (entity, render_entity, terrain_chunk, changed) in &query {
        commands.entity(render_entity.id());

        let main_entity = MainEntity::from(entity);
        active_entities.insert(main_entity);

        // Determine if we need to create or update
        let needs_update = changed.is_some() || !data_store.contains_key(&main_entity);

        // Skip if no update needed
        if !needs_update {
            continue;
        }

        // Convert terrain data to texture format
        let format_size = size_of::<TerrainCell>();

        // Update existing texture if data exists, otherwise create a new one
        if changed.is_some() && data_store.contains_key(&main_entity) {
            // Get existing GPU image
            if let Some(data) = data_store.get_mut(&main_entity) {
                // Update existing texture
                terrain_chunk.provide_texture(|image_data| {
                    render_queue.write_texture(
                        data.terrain_map_gpu_image.texture.as_image_copy(),
                        &image_data,
                        ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(ISLAND_CHUNK_SIZE * format_size as u32),
                            rows_per_image: None,
                        },
                        Extent3d {
                            width: ISLAND_CHUNK_SIZE,
                            height: ISLAND_CHUNK_SIZE,
                            depth_or_array_layers: 1,
                        },
                    );
                });

                trace!(
                    "Updated existing terrain chunk texture for entity: {:?}",
                    entity
                );
                continue;
            }
        }

        // Create texture descriptor
        let texture_descriptor = TextureDescriptor {
            label: Some("terrain_chunk_texture"),
            size: Extent3d {
                width: ISLAND_CHUNK_SIZE,
                height: ISLAND_CHUNK_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        };

        // Create GPU image
        let gpu_image = {
            // Create texture
            let texture = render_device.create_texture(&texture_descriptor);

            // Create sampler
            let sampler = render_device.create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                ..Default::default()
            });

            // Write texture data to GPU
            let format_size = size_of::<TerrainCell>();
            terrain_chunk.provide_texture(|image_data| {
                render_queue.write_texture(
                    texture.as_image_copy(),
                    &image_data,
                    ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(ISLAND_CHUNK_SIZE * format_size as u32),
                        rows_per_image: None,
                    },
                    texture_descriptor.size,
                );
            });

            // Create texture view
            let texture_view = texture.create_view(&TextureViewDescriptor::default());

            // Create GpuImage
            GpuImage {
                texture,
                texture_view,
                texture_format: texture_descriptor.format,
                sampler,
                size: UVec2::new(ISLAND_CHUNK_SIZE, ISLAND_CHUNK_SIZE),
                mip_level_count: texture_descriptor.mip_level_count,
            }
        };

        // Store in data store
        data_store.insert(
            entity.into(),
            TerrainPerChunkData {
                terrain_map_gpu_image: gpu_image,
            },
        );
    }

    // Clean up data for entities that no longer exist
    let mut to_remove = Vec::new();
    for entity_key in data_store.keys() {
        if !active_entities.contains(entity_key) {
            to_remove.push(*entity_key);
        }
    }

    for entity_key in to_remove {
        trace!(
            "Removing terrain chunk data for destroyed entity: {:?}",
            entity_key
        );
        data_store.remove(&entity_key);
    }
}

pub fn prepare_terrain_bind_group<M: TerrainMaterial + 'static>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<TerrainPipeline<M>>,
    data_store: Res<TerrainPerChunkDataStore>,
    versions: ResMut<TerrainRendererVersions>,
    query: Query<(Entity, &MainEntity, &TerrainChunkRenderer)>,
) {
    trace!(
        "Preparing terrain bind groups for {} entities",
        query.iter().count()
    );

    let versions = &mut versions.into_inner().0;

    for (entity, main_entity, renderer) in query.iter() {
        if let Some(last_version) = versions.get(main_entity) {
            if *last_version == renderer.version {
                continue;
            }
        }

        let prepare_chunk = |index: usize| {
            let chunk_entity = renderer.chunks[index];
            info!(
                "renderer({}).chunks[{}] = {:?}",
                main_entity.id(),
                index,
                chunk_entity
            );
            if let Some(data) = data_store.0.get(&MainEntity::from(chunk_entity)) {
                data.terrain_map_gpu_image.texture_view.into_binding()
            } else {
                pipeline.dummy_black_gpu_image.texture_view.into_binding()
            }
        };

        let bind_group = render_device.create_bind_group(
            "terrain_per_chunk_data",
            &pipeline.per_chunk_data_layout,
            &BindGroupEntries::sequential((
                prepare_chunk(0),
                prepare_chunk(1),
                prepare_chunk(2),
                prepare_chunk(3),
                pipeline.dummy_black_gpu_image.sampler.into_binding(),
            )),
        );

        commands
            .entity(entity)
            .insert(TerrainPerChunkBindGroup { value: bind_group });

        versions.insert(main_entity.clone(), renderer.version);
    }
}

impl<M: TerrainMaterial> FromWorld for TerrainPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        trace!("Creating TerrainPipeline from world");
        let mesh2d_pipeline = Mesh2dPipeline::from_world(world);
        let terrain_shader: Handle<Shader> = match M::fragment_shader() {
            ShaderRef::Default => {
                panic!("fragment shader required!");
            }
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => world.load_asset(path),
        };

        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<RenderQueue>,
            Res<DefaultImageSampler>,
        )> = SystemState::new(world);

        let (render_device, render_queue, default_sampler) = system_state.get_mut(world);
        let render_device = render_device.into_inner();
        let dummy_black_gpu_image = {
            let image = Image::default();
            let texture = render_device.create_texture(&image.texture_descriptor);
            let sampler = match image.sampler {
                ImageSampler::Default => (**default_sampler).clone(),
                ImageSampler::Descriptor(ref descriptor) => {
                    render_device.create_sampler(&descriptor.as_wgpu())
                }
            };

            let format_size = image.texture_descriptor.format.pixel_size();
            render_queue.write_texture(
                texture.as_image_copy(),
                &image.data,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(image.width() * format_size as u32),
                    rows_per_image: None,
                },
                image.texture_descriptor.size,
            );

            let texture_view = texture.create_view(&TextureViewDescriptor::default());
            GpuImage {
                texture,
                texture_view,
                texture_format: image.texture_descriptor.format,
                sampler,
                size: image.size(),
                mip_level_count: image.texture_descriptor.mip_level_count,
            }
        };

        Self {
            mesh2d_pipeline,
            material_layout: M::bind_group_layout(render_device),
            per_chunk_data_layout: render_device.create_bind_group_layout(
                "per_chunk_data",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::FRAGMENT,
                    (
                        (0, texture_2d(TextureSampleType::Float { filterable: true })),
                        (1, texture_2d(TextureSampleType::Float { filterable: true })),
                        (2, texture_2d(TextureSampleType::Float { filterable: true })),
                        (3, texture_2d(TextureSampleType::Float { filterable: true })),
                        (4, sampler(SamplerBindingType::Filtering)),
                    ),
                ),
            ),
            material_bind_group: None,
            dummy_black_gpu_image,
            terrain_shader,
            terrain_material: M::from_world(world),
        }
    }
}

impl<M: TerrainMaterial> SpecializedMeshPipeline for TerrainPipeline<M> {
    type Key = Mesh2dPipelineKey;
    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        trace!("Specializing TerrainPipeline with key: {:?}", key);
        let mut descriptor = self.mesh2d_pipeline.specialize(key, layout)?;
        descriptor.fragment.as_mut().unwrap().shader = self.terrain_shader.clone();
        descriptor.layout = vec![
            self.mesh2d_pipeline.view_layout.clone(),
            self.mesh2d_pipeline.mesh_layout.clone(),
            self.material_layout.clone(),
            self.per_chunk_data_layout.clone(),
        ];

        Ok(descriptor)
    }
}

/// Base terrain material that determines physical properties
#[derive(Debug, PartialEq, FromRepr)]
#[repr(u8)]
pub enum TerrainBase {
    Rock = 0,
    Sand = 1,
    Soil = 2,
    Mud = 3,
}

impl Default for TerrainBase {
    fn default() -> Self {
        TerrainBase::Rock
    }
}

/// Surface covering that determines visual appearance
#[derive(Debug, PartialEq, FromRepr)]
#[repr(u8)]
pub enum TerrainSurface {
    Bare = 0,
    Water = 1,
    Grass = 2,
    Snow = 3,
}

impl Default for TerrainSurface {
    fn default() -> Self {
        TerrainSurface::Bare
    }
}

#[derive(Debug, Default, Copy, Zeroable, Clone, Pod, Reflect)]
#[repr(C)]
pub struct TerrainCell {
    height: u8,
    base_type: u8,
    surface_type: u8,
    _padding: u8,
}

/// 标记地形块已更改，需要更新GPU纹理数据
#[derive(Component, Clone, Debug, Default)]
#[component(storage = "SparseSet")]
pub struct TerrainChunkChanged;

#[derive(Component, Clone, Debug, Reflect)]
pub struct TerrainChunk {
    pos: IVec2,
    data: Vec<TerrainCell>,
}

pub trait AsTextureProvider {
    fn provide_texture<F>(&self, callback: F) where F: Fn(&[u8]);
}

impl AsTextureProvider for TerrainChunk {
    fn provide_texture<F>(&self, callback: F) where F: Fn(&[u8]) {
        callback(bytemuck::cast_slice(&self.data));
    }
}

pub trait GetChunkPos {
    fn get_pos(&self) -> IVec2;
}

impl GetChunkPos for TerrainChunk {
    fn get_pos(&self) -> IVec2 { return self.pos; }
}

#[derive(Component, Clone, Debug, Reflect, ExtractComponent)]
pub struct TerrainChunkRenderer {
    pos: IVec2,
    chunks: [Entity; 4],
    version: usize,
}

impl TerrainChunkRenderer {
    pub fn mark_dirty(&mut self) {
        self.version += 1;
    }
}

/// Generates terrain data for a chunk using multiple noise sources
///
/// # Parameters
/// * `chunk_data` - Vector to populate with terrain cell data
/// * `noise_source` - Primary noise source for height generation
pub fn generate_tiles(
    id: IVec2,
    chunk_data: &mut Vec<TerrainCell>,
    noise_source: &NoiseSource<Frequency<Fbm<OpenSimplex2>, Constant>>,
) {
    // let center = vec2( ISLAND_CHUNK_SIZE as f32 / 2.0f32, ISLAND_CHUNK_SIZE as f32 / 2.0f32, );
    let center = vec2(0.0, 0.0);

    // Create specialized noise sources for different terrain features
    let base_noise = OpenSimplex2
        .fbm(3, 0.6, 2.2)
        .frequency(1.8 / ISLAND_CHUNK_SIZE as f32);

    // Create a noise source for island shape irregularity
    let shape_noise = OpenSimplex2
        .fbm(4, 0.7, 2.0)
        .frequency(3.0 / ISLAND_CHUNK_SIZE as f32);

    let start_x = id.x * ISLAND_CHUNK_SIZE as i32;
    let start_y = id.y * ISLAND_CHUNK_SIZE as i32;

    chunk_data.par_chunk_map_mut(
        ComputeTaskPool::get(),
        ISLAND_CHUNK_SIZE as usize,
        |chunk_index, data| {
            for (i, item) in data.iter_mut().enumerate() {
                let index = (chunk_index * ISLAND_CHUNK_SIZE as usize + i) as u32;
                let ux = index % ISLAND_CHUNK_SIZE;
                let uy = ISLAND_CHUNK_SIZE - index / ISLAND_CHUNK_SIZE;
                let x = ux as i32 + start_x;
                let y = uy as i32 + start_y;
                let position = vec2(x as f32, y as f32);

                // Height noise
                let noise = noise_source
                    .noise
                    .sample_with_seed([position.x, position.y], RANDOM_SEED as i32)
                    * 0.5
                    + 0.5;

                // Get shape noise to create irregular island shape
                let shape_noise_value = shape_noise
                    .sample_with_seed([position.x, position.y], (RANDOM_SEED + 123) as i32)
                    * 0.5
                    + 0.5;

                // Perturb the distance calculation with noise to create irregular coastline
                let angle = position.y.atan2(position.x);
                let noise_factor = 0.3; // Controls how irregular the coastline is
                let distance_perturbation = (shape_noise_value - 0.5) * noise_factor;

                // Calculate distance with perturbation
                let distance_from_center =
                    (position - center).length() * (1.0 + distance_perturbation);

                // Use a different exponent to create more varied island shape
                let height_scale = 1.0 - (distance_from_center / ISLAND_RADIUS).powf(1.8);

                // Apply additional shape variation based on angle
                let angular_variation = (angle * 4.0).sin() * 0.1;
                let height_scale = (height_scale + angular_variation).max(0.0);

                let height = height_scale * noise * HEIGHT_LIMIT;

                // Base type noise
                let base_noise_value = base_noise
                    .sample_with_seed([position.x, position.y], (RANDOM_SEED + 42) as i32)
                    * 0.5
                    + 0.5;

                // Determine if this is a coastal area (near sea level)
                let is_coastal = height > SEA_LEVEL - 5.0 && height < SEA_LEVEL + 10.0;

                // Determine base type
                let base_type = if is_coastal {
                    // Coastal areas are always sandy beaches
                    TerrainBase::Sand
                } else if height <= SEA_LEVEL {
                    // Underwater areas - mix of sand and mud
                    if base_noise_value < 0.4 {
                        TerrainBase::Sand
                    } else {
                        TerrainBase::Mud
                    }
                } else {
                    // Land areas - mix of rock and soil based on noise and height
                    if height > SEA_LEVEL + 50.0 || base_noise_value > 0.7 {
                        TerrainBase::Rock // Higher elevations and some random areas are rocky
                    } else {
                        TerrainBase::Soil // Most land is soil
                    }
                };

                // Determine surface type
                let surface_type = if height <= SEA_LEVEL {
                    // Underwater is always water
                    TerrainSurface::Water
                } else {
                    // Land areas - determine surface based on height and noise
                    let elevation_percent = (height - SEA_LEVEL) / (HEIGHT_LIMIT - SEA_LEVEL);

                    // Use base_noise to add variation to surface type boundaries
                    let surface_noise = base_noise_value;

                    if elevation_percent > 0.7 || (elevation_percent > 0.6 && surface_noise > 0.6) {
                        // High elevations get snow
                        TerrainSurface::Snow
                    } else if elevation_percent < 0.3
                        || (elevation_percent < 0.4 && surface_noise < 0.4)
                    {
                        // Lower elevations get grass (except beaches which are handled by base type)
                        if !is_coastal {
                            TerrainSurface::Grass
                        } else {
                            TerrainSurface::Bare // Beaches remain bare
                        }
                    } else {
                        // Middle elevations are bare
                        TerrainSurface::Bare
                    }
                };

                // Set cell data
                item.height = height as u8;
                item.base_type = base_type as u8;
                item.surface_type = surface_type as u8;
            }
        },
    );
}

pub fn spawn_initial_chunks(
    mut commands: Commands,
    noise_source: Res<NoiseSource<Frequency<Fbm<OpenSimplex2>, Constant>>>,
) {
    let noise_source = noise_source.into_inner();
    spawn_tilemap(&mut commands, noise_source, IVec2::new(0, 0));
    spawn_tilemap(&mut commands, noise_source, IVec2::new(-1, 0));
    spawn_tilemap(&mut commands, noise_source, IVec2::new(-1, -1));
    spawn_tilemap(&mut commands, noise_source, IVec2::new(0, -1));
}

pub fn spawn_tilemap(
    commands: &mut Commands,
    noise_source: &NoiseSource<Frequency<Fbm<OpenSimplex2>, Constant>>,
    id: IVec2,
) {
    trace!("Spawning tilemap with chunk size: {}", ISLAND_CHUNK_SIZE);
    let mut chunk_data =
        vec![TerrainCell::default(); (ISLAND_CHUNK_SIZE * ISLAND_CHUNK_SIZE) as usize];

    generate_tiles(id, &mut chunk_data, noise_source);
    commands.spawn((TerrainChunk {
        pos: id,
        data: chunk_data,
    },));
}

#[derive(Resource, Default, Debug, Reflect)]
pub struct TerrainChunkRenderers(HashMap<IVec2, Entity>);

#[derive(Resource)]
pub struct TerrainChunkMesh(Handle<Mesh>);

impl FromWorld for TerrainChunkMesh {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.get_resource_mut::<Assets<Mesh>>().unwrap();
        let handle = meshes.add(Rectangle::new(
            ISLAND_CHUNK_SIZE as f32,
            ISLAND_CHUNK_SIZE as f32,
        ));
        Self(handle)
    }
}

pub fn on_add_chunk<T: Component + GetChunkPos>(
    trigger: Trigger<OnAdd, T>,
    query_chunks: Query<&T>,
    mut commands: Commands,
    renderers: ResMut<TerrainChunkRenderers>,
    chunk_mesh: Res<TerrainChunkMesh>,
) {
    let renderers = renderers.into_inner();
    let entity = trigger.entity();
    if let Ok(chunk) = query_chunks.get(entity) {
        let chunk_pos = chunk.get_pos();
        let positions = vec![
            chunk_pos,
            chunk_pos + IVec2::new(1, 0),
            chunk_pos + IVec2::new(0, 1),
            chunk_pos + IVec2::new(1, 1),
        ];

        for (i, pos) in positions.iter().enumerate() {
            if let Some(renderer_entity) = renderers.0.get(pos) {
                if renderer_entity.index() != Entity::PLACEHOLDER.index() {
                    let renderer_entity = *renderer_entity;
                    commands
                        .entity(renderer_entity)
                        .entry::<TerrainChunkRenderer>()
                        .and_modify(move |mut renderer| {
                            renderer.chunks[i] = entity;
                            renderer.mark_dirty();
                            info!(
                                "set renderer {} chunks [{}] to {} ",
                                renderer_entity, i, entity
                            );
                        });
                } else {
                    warn!("renderer entity {} not ready!", renderer_entity);
                }
            } else {
                let mut chunks = [Entity::PLACEHOLDER; 4];
                chunks[i] = entity;

                let id = commands
                    .spawn((
                        Mesh2d(chunk_mesh.0.clone()),
                        NoAutomaticBatching,
                        TerrainChunkRenderer {
                            pos: *pos,
                            chunks,
                            version: 0,
                        },
                        Transform::from_translation(vec3(
                            (pos.x * ISLAND_CHUNK_SIZE as i32) as f32,
                            (pos.y * ISLAND_CHUNK_SIZE as i32) as f32,
                            0.0,
                        )),
                    ))
                    .id();

                info!("new renderer {} chunks [{}] to {} ", id, i, entity);
                renderers.0.insert(*pos, id);
            }
        }
    }
}

pub fn on_remove_chunk<T: Component + GetChunkPos>(
    trigger: Trigger<OnAdd, T>,
    query_chunks: Query<&T>,
    mut query_renderers: Query<&mut TerrainChunkRenderer>,
    renderers: ResMut<TerrainChunkRenderers>,
) {
    let renderers = renderers.into_inner();
    let entity = trigger.entity();
    if let Ok(chunk) = query_chunks.get(entity) {
        let chunk_pos = chunk.get_pos();
        let positions = vec![
            chunk_pos,
            chunk_pos + IVec2::new(1, 0),
            chunk_pos + IVec2::new(0, 1),
            chunk_pos + IVec2::new(1, 1),
        ];
        for (i, pos) in positions.iter().enumerate() {
            if let Some(entity) = renderers.0.get(pos) {
                if let Ok(mut renderer) = query_renderers.get_mut(*entity) {
                    renderer.chunks[i] = Entity::PLACEHOLDER;
                }
            }
        }
    }
}

fn switch_render_mode(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut render_mode: ResMut<TerrainRenderMode>,
) {
    trace!(
        "Checking for render mode switch, current mode: {}",
        render_mode.mode
    );
    if keyboard_input.just_pressed(KeyCode::Minus) {
        render_mode.mode = (render_mode.mode + 1) % MAX_RENDER_MODE;
        info!("Switched terrain render mode to: {}", render_mode.mode);
    }
}

// Update the material uniforms including render mode and time values
fn update_material_uniforms(
    time: Res<Time>,
    render_mode: Res<TerrainRenderMode>,
    terrain_pipeline: ResMut<TerrainPipeline<TerrainBaseMaterial>>,
    render_queue: Res<RenderQueue>,
) {
    trace!(
        "Updating terrain material with render mode: {}",
        render_mode.mode
    );

    let terrain_pipeline = terrain_pipeline.into_inner();
    let material = &mut terrain_pipeline.terrain_material;

    // Update all uniform fields
    material.material_uniform.mode = render_mode.mode;

    // 如果material_bind_group已存在，更新其中的uniform buffer
    if let Some(prepared_bind_group) = &mut terrain_pipeline.material_bind_group {
        for (binding_index, binding_resource) in &prepared_bind_group.bindings {
            if *binding_index == 0 {
                if let OwnedBindingResource::Buffer(buffer) = binding_resource {
                    render_queue.write_buffer(
                        buffer,
                        0, // 偏移量为0
                        bytemuck::cast_slice(&[material.material_uniform]),
                    );
                    break;
                }
            }
        }
    }
}

pub fn extract_terrain_mesh2d(
    query: Extract<
        Query<(Entity, &ViewVisibility, &GlobalTransform, &Mesh2d), With<TerrainChunkRenderer>>,
    >,
    mut render_mesh_instances: ResMut<RenderTerrainMeshInstances>,
) {
    trace!(
        "Extracting terrain renderer mesh2d instances from {} entities",
        query.iter().count()
    );
    for (entity, view_visibility, transform, handle) in &query {
        if !view_visibility.get() {
            continue;
        }

        let transforms = Mesh2dTransforms {
            world_from_local: (&transform.affine()).into(),
            flags: MeshFlags::empty().bits(),
        };

        render_mesh_instances.insert(
            entity.into(),
            RenderMesh2dInstance {
                mesh_asset_id: handle.0.id(),
                transforms,
                material_bind_group_id: Material2dBindGroupId::default(),
                automatic_batching: false,
                // tag: 0, // For next version
            },
        );
    }
}

pub fn queue_terrain_mesh2d<M: TerrainMaterial + 'static>(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    terrain_pipeline: Res<TerrainPipeline<M>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<TerrainPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderTerrainMeshInstances>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(Entity, &RenderVisibleEntities, &ExtractedView, &Msaa)>,
) {
    trace!(
        "Queueing terrain mesh2d with {} instances",
        render_mesh_instances.len()
    );

    let terrain_pipeline = terrain_pipeline.into_inner();

    if render_mesh_instances.is_empty() {
        return;
    }

    // Iterate each view (a camera is a view)
    for (view_entity, visible_entities, view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };

        let draw_terrain_mesh2d = transparent_draw_functions.read().id::<DrawTerrainBaseMesh2d>();

        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr);

        // Queue all entities visible to that view
        for (render_entity, visible_entity) in visible_entities.iter::<With<Mesh2d>>() {
            if let Some(mesh_instance) = render_mesh_instances.get(visible_entity) {
                let mesh2d_handle = mesh_instance.mesh_asset_id;
                let mesh2d_transforms = &mesh_instance.transforms;
                // Get our specialized pipeline
                let mut mesh2d_key = mesh_key;
                let Some(mesh) = render_meshes.get(mesh2d_handle) else {
                    continue;
                };
                mesh2d_key |= Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology());

                let pipeline_id = pipelines.specialize(
                    &pipeline_cache,
                    &terrain_pipeline,
                    mesh2d_key,
                    &mesh.layout,
                );

                let pipeline_id = match pipeline_id {
                    Ok(id) => id,
                    Err(err) => {
                        error!("{}", err);
                        continue;
                    }
                };

                let mesh_z = mesh2d_transforms.world_from_local.translation.z;
                transparent_phase.add(Transparent2d {
                    entity: (*render_entity, *visible_entity),
                    draw_function: draw_terrain_mesh2d,
                    pipeline: pipeline_id,
                    // The 2d render items are sorted according to their z value before rendering,
                    // in order to get correct transparency
                    sort_key: FloatOrd(mesh_z),
                    // This material is not batched
                    batch_range: 0..1,
                    extra_index: PhaseItemExtraIndex::NONE,
                });
            }
        }
    }
}

pub(crate) struct Terrain2dPlugin;

impl Plugin for Terrain2dPlugin {
    fn build(&self, app: &mut App) {
        trace!("Building Terrain2dPlugin");
        let fbm = OpenSimplex2
            .fbm(4, 0.5, 2.0)
            .frequency(2.4 / ISLAND_RADIUS);

        app.add_plugins(SyncComponentPlugin::<TerrainChunk>::default());
        app.add_plugins(ExtractResourcePlugin::<TerrainRenderMode>::default());
        app.add_plugins(ExtractComponentPlugin::<TerrainChunkRenderer>::default());
        app.register_type::<TerrainChunkRenderer>();

        app.insert_resource(NoiseSource { noise: fbm })
            .init_resource::<TerrainChunkMesh>()
            .init_resource::<TerrainChunkRenderers>()
            .init_resource::<TerrainRenderMode>()
            .add_systems(Startup, spawn_initial_chunks)
            .add_systems(Update, switch_render_mode)
            .add_observer(on_add_chunk::<TerrainChunk>)
            .add_observer(on_remove_chunk::<TerrainChunk>);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent2d, DrawTerrainBaseMesh2d>()
                .init_resource::<SpecializedMeshPipelines<TerrainPipeline<TerrainBaseMaterial>>>()
                .init_resource::<TerrainRendererVersions>()
                .init_resource::<RenderTerrainMeshInstances>()
                .init_resource::<TerrainPerChunkDataStore>()
                .init_resource::<TerrainRenderMode>()
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_terrain_chunk::<TerrainChunk>,
                        extract_terrain_mesh2d.after(extract_mesh2d),
                    ),
                )
                .add_systems(
                    Render,
                    (
                        update_material_uniforms,
                        bind_terrain_material.in_set(RenderSet::PrepareBindGroups),
                        prepare_terrain_bind_group::<TerrainBaseMaterial>
                            .in_set(RenderSet::PrepareBindGroups),
                        queue_terrain_mesh2d::<TerrainBaseMaterial>.in_set(RenderSet::QueueMeshes),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        trace!("Finishing Terrain2dPlugin setup");
        app.get_sub_app_mut(RenderApp)
            .unwrap()
            .init_resource::<TerrainPipeline<TerrainBaseMaterial>>();
    }
}

pub fn bind_terrain_material(
    mut pipeline: ResMut<TerrainPipeline<TerrainBaseMaterial>>,
    render_device: Res<RenderDevice>,
    mut param: StaticSystemParam<<TerrainBaseMaterial as AsBindGroup>::Param>,
) {
    trace!("Binding terrain material");
    // Already prepared
    if pipeline.material_bind_group.is_some() {
        return;
    }

    let terrain_material = &pipeline.terrain_material;

    let Ok(prepared) =
        terrain_material.as_bind_group(&pipeline.material_layout, &render_device, &mut param)
    else {
        return;
    };

    pipeline.material_bind_group = Some(prepared);
}
