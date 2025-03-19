use crate::{NoiseSource, HEIGHT_LIMIT, RANDOM_SEED, SEA_LEVEL};
use bevy::asset::Assets;
use bevy::core_pipeline::core_2d::Transparent2d;
use bevy::ecs::system::lifetimeless::{Read, SRes};
use bevy::ecs::system::{StaticSystemParam, SystemParamItem};
use bevy::image::{Image, ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
use bevy::math::{vec2, FloatOrd};
use bevy::pbr::MeshFlags;
use bevy::prelude::*;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{MeshVertexBufferLayoutRef, RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d};
use bevy::render::render_resource::{
    AddressMode, AsBindGroup, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
    Extent3d, FilterMode, ImageDataLayout, IntoBinding, PipelineCache, PreparedBindGroup,
    RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderStages,
    SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDescriptor,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::sync_world::{MainEntity, MainEntityHashMap, RenderEntity, SyncToRenderWorld};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, RenderVisibleEntities};
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::sprite::{
    extract_mesh2d, Material2dBindGroupId, Mesh2dBindGroup, Mesh2dPipeline, Mesh2dPipelineKey,
    Mesh2dTransforms, RenderMesh2dInstance, SetMesh2dViewBindGroup,
};
use bevy::tasks::{ComputeTaskPool, ParallelSliceMut};
use bytemuck::{Pod, Zeroable};
use noise_functions::modifiers::{Fbm, Frequency};
use noise_functions::{Constant, Noise, OpenSimplex2, Sample};
use std::mem::size_of;
use bevy::ecs::query::ROQueryItem;
use bevy::render::sync_component::SyncComponentPlugin;
use strum::FromRepr;

const ISLAND_CHUNK_SIZE: u32 = 256; // We will finally reach 512

pub const TILE_PIXEL_SIZE: f32 = 32.0;

const MAX_RENDER_MODE: u32 = 2;

#[derive(Resource, Default)]
pub struct TerrainRenderMode {
    pub mode: u32,
}

const TERRAIN_SHADER_PATH: &str = "shaders/terrain_base.wgsl";
const SUPER_PERLIN_TEXTURE_PATH: &str = "textures/sperlin_rock.png";
const GRAINY_TEXTURE_PATH: &str = "textures/grainy.png";

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub(crate) struct TerrainMaterial {
    #[uniform(0)]
    mode: u32,

    #[texture(1)]
    #[sampler(2)]
    super_perlin_rock: Option<Handle<Image>>,

    #[texture(3)]
    #[sampler(4)]
    grainy_texture: Option<Handle<Image>>,
}

#[derive(Resource)]
pub struct TerrainPipeline {
    /// This pipeline wraps the standard [`Mesh2dPipeline`]
    mesh2d_pipeline: Mesh2dPipeline,
    material_layout: BindGroupLayout,
    per_chunk_data_layout: BindGroupLayout,
    material_bind_group: Option<PreparedBindGroup<<TerrainMaterial as AsBindGroup>::Data>>,
    terrain_shader: Handle<Shader>,
    terrain_material: Handle<TerrainMaterial>,
}

pub struct TerrainPerChunkData {
    terrain_map_gpu_image: GpuImage,
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct TerrainPerChunkDataStore(MainEntityHashMap<TerrainPerChunkData>);

#[derive(Component, Debug, Clone)]
pub struct TerrainPerChunkBindGroup {
    pub value: BindGroup,
}

pub fn extract_terrain_chunk(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut data_store: ResMut<TerrainPerChunkDataStore>,
    query: Extract<Query<(Entity, &RenderEntity, &TerrainChunk)>>,
) {
    trace!("Extracting terrain chunks");
    for (entity, render_entity, terrain_chunk) in &query {
        commands.entity(render_entity.id()).insert(TerrainChunkGpuMarker{});

        // Skip if we already have data for this chunk
        if data_store.contains_key(&MainEntity::from(entity)) {
            continue;
        }

        // Convert terrain data to texture format
        let image_data: Vec<u8> = bytemuck::cast_slice(&terrain_chunk.data).to_vec();

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

        // Create GPU image directly
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
}

pub fn prepare_terrain_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<TerrainPipeline>,
    data_store: Res<TerrainPerChunkDataStore>,
    query: Query<(Entity, &MainEntity), With<TerrainChunkGpuMarker>>,
) {
    trace!("Preparing terrain bind groups for {} entities", query.iter().count());
    for (entity, main_entity) in query.iter() {
        if let Some(data) = data_store.get(&MainEntity::from(main_entity.id())) {
            let bind_group = render_device.create_bind_group(
                "terrain_per_chunk",
                &pipeline.per_chunk_data_layout,
                &BindGroupEntries::with_indices((
                    (0, data.terrain_map_gpu_image.texture_view.into_binding()),
                    (1, data.terrain_map_gpu_image.sampler.into_binding()),
                )),
            );
            
            commands.entity(entity).insert(TerrainPerChunkBindGroup { value: bind_group });
        }
    }
}

impl FromWorld for TerrainPipeline {
    fn from_world(world: &mut World) -> Self {
        trace!("Creating TerrainPipeline from world");
        let mesh2d_pipeline = Mesh2dPipeline::from_world(world);
        let render_device = world.resource::<RenderDevice>();

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

        Self {
            mesh2d_pipeline,
            material_layout: TerrainMaterial::bind_group_layout(render_device),
            per_chunk_data_layout: render_device.create_bind_group_layout(
                "per_chunk_data",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::FRAGMENT,
                    (
                        (0, texture_2d(TextureSampleType::Float { filterable: true })),
                        (1, sampler(SamplerBindingType::Filtering)),
                    ),
                ),
            ),
            material_bind_group: None,
            terrain_shader: world.load_asset(TERRAIN_SHADER_PATH),
            terrain_material: world.add_asset(TerrainMaterial {
                mode: 0,
                super_perlin_rock: Some(world.load_asset_with_settings(SUPER_PERLIN_TEXTURE_PATH, image_repeat_settings)),
                grainy_texture: Some(world.load_asset_with_settings(GRAINY_TEXTURE_PATH, image_repeat_settings)),
            }),
        }
    }
}

impl SpecializedMeshPipeline for TerrainPipeline {
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

#[derive(Resource, Deref, DerefMut, Default)]
pub struct RenderTerrainMeshInstances(MainEntityHashMap<RenderMesh2dInstance>);

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

#[derive(Debug, Default, Copy, Zeroable, Clone, Pod)]
#[repr(C)]
pub struct TerrainCell {
    height: u8,
    base_type: u8,
    surface_type: u8,
    _padding: u8,
}

#[derive(Component, Clone, Debug)]
pub struct TerrainChunkGpuMarker;

#[derive(Component, Clone, Debug)]
pub struct TerrainChunk {
    center_position: Vec2,
    data: Vec<TerrainCell>,
}

pub fn spawn_tilemap(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    noise_source: Res<NoiseSource<Frequency<Fbm<OpenSimplex2>, Constant>>>,
) {
    trace!("Spawning tilemap with chunk size: {}", ISLAND_CHUNK_SIZE);
    let mut chunk_data =
        vec![TerrainCell::default(); (ISLAND_CHUNK_SIZE * ISLAND_CHUNK_SIZE) as usize];

    // let mut image_data = vec![0u8; (ISLAND_CHUNK_SIZE * ISLAND_CHUNK_SIZE) as usize];
    let center = vec2(
        ISLAND_CHUNK_SIZE as f32 / 2.0f32,
        ISLAND_CHUNK_SIZE as f32 / 2.0f32,
    );
    
    // Create a second noise source for terrain base type with different frequency
    let base_noise = OpenSimplex2
        .fbm(3, 0.6, 2.2)
        .frequency(1.8 / ISLAND_CHUNK_SIZE as f32);
        
    // Create a noise source for island shape irregularity
    let shape_noise = OpenSimplex2
        .fbm(4, 0.7, 2.0)
        .frequency(3.0 / ISLAND_CHUNK_SIZE as f32);

    chunk_data.par_chunk_map_mut(
        ComputeTaskPool::get(),
        ISLAND_CHUNK_SIZE as usize,
        |chunk_index, data| {
            for (i, item) in data.iter_mut().enumerate() {
                let index = chunk_index * ISLAND_CHUNK_SIZE as usize + i;
                let x = index % (ISLAND_CHUNK_SIZE as usize);
                let y = index / (ISLAND_CHUNK_SIZE as usize);
                let position = vec2(x as f32, y as f32) - center;
                
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
                let distance_from_center = position.length() * (1.0 + distance_perturbation);
                
                // Use a different exponent to create more varied island shape
                let height_scale = 1.0 - (distance_from_center / (ISLAND_CHUNK_SIZE as f32 * 0.5)).powf(1.8);
                
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
                let is_coastal = height > SEA_LEVEL - 5.0 && height < SEA_LEVEL + 5.0;
                
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
                    } else if elevation_percent < 0.3 || (elevation_percent < 0.4 && surface_noise < 0.4) {
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

                item.height = height as u8;
                item.base_type = base_type as u8;
                item.surface_type = surface_type as u8;
            }
        },
    );

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(
            ISLAND_CHUNK_SIZE as f32,
            ISLAND_CHUNK_SIZE as f32,
        ))),
        TerrainChunk {
            center_position: vec2(0.0, 0.0),
            data: chunk_data,
        },
    ));
}

fn switch_render_mode(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut render_mode: ResMut<TerrainRenderMode>,
) {
    trace!("Checking for render mode switch, current mode: {}", render_mode.mode);
    if keyboard_input.just_pressed(KeyCode::Minus) {
        render_mode.mode = (render_mode.mode + 1) % MAX_RENDER_MODE;
        info!("Switched terrain render mode to: {}", render_mode.mode);
    }
}

// Extract the render mode to the render world
fn extract_terrain_render_mode(
    render_mode: Extract<Res<TerrainRenderMode>>,
    mut render_app_render_mode: ResMut<TerrainRenderMode>,
) {
    trace!("Extracting terrain render mode: {}", render_mode.mode);
    render_app_render_mode.mode = render_mode.mode;
}

// Update the material in the render world based on the extracted render mode
fn update_terrain_material(
    render_mode: Res<TerrainRenderMode>,
    terrain_pipeline: Res<TerrainPipeline>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    trace!("Updating terrain material with render mode: {}", render_mode.mode);
    if let Some(material) = materials.get_mut(terrain_pipeline.terrain_material.id()) {
        material.mode = render_mode.mode;
    }
}

pub fn extract_terrain_mesh2d(
    query: Extract<
        Query<(
            Entity,
            &ViewVisibility,
            &GlobalTransform,
            &Mesh2d,
        ),
        With<TerrainChunk>>,
    >,
    mut render_mesh_instances: ResMut<RenderTerrainMeshInstances>,
) {
    trace!("Extracting terrain mesh2d instances from {} entities", query.iter().count());
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

pub fn queue_terrain_mesh2d(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    terrain_pipeline: Res<TerrainPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<TerrainPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderTerrainMeshInstances>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(Entity, &RenderVisibleEntities, &ExtractedView, &Msaa)>,
) {
    trace!("Queueing terrain mesh2d with {} instances", render_mesh_instances.len());
    if render_mesh_instances.is_empty() {
        return;
    }

    // Iterate each view (a camera is a view)
    for (view_entity, visible_entities, view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };

        let draw_terrain_mesh2d = transparent_draw_functions.read().id::<DrawTerrainMesh2d>();

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
            .frequency(2.4 / ISLAND_CHUNK_SIZE as f32);

        app.add_plugins(SyncComponentPlugin::<TerrainChunk>::default());

        app.insert_resource(NoiseSource { noise: fbm })
            .init_resource::<TerrainRenderMode>()
            .add_systems(Startup, spawn_tilemap)
            .add_systems(Update, switch_render_mode);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent2d, DrawTerrainMesh2d>()
                .init_resource::<SpecializedMeshPipelines<TerrainPipeline>>()
                .init_resource::<RenderTerrainMeshInstances>()
                .init_resource::<Assets<TerrainMaterial>>()
                .init_resource::<TerrainPerChunkDataStore>()
                .init_resource::<TerrainRenderMode>()
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_terrain_chunk,
                        extract_terrain_render_mode,
                        extract_terrain_mesh2d.after(extract_mesh2d),
                    ),
                )
                .add_systems(
                    Render,
                    (
                        update_terrain_material.in_set(RenderSet::PrepareBindGroups),
                        bind_terrain_material.in_set(RenderSet::PrepareBindGroups),
                        prepare_terrain_bind_group.in_set(RenderSet::PrepareBindGroups),
                        queue_terrain_mesh2d.in_set(RenderSet::QueueMeshes),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        trace!("Finishing Terrain2dPlugin setup");
        app.get_sub_app_mut(RenderApp)
            .unwrap()
            .init_resource::<TerrainPipeline>();
    }
}

pub struct DrawTerrain2d;
impl<P: PhaseItem> RenderCommand<P> for DrawTerrain2d {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderTerrainMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        (meshes, terrain_mesh_instances, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        trace!("Rendering DrawTerrain2d for entity: {:?}", item.main_entity());
        let meshes = meshes.into_inner();
        let terrain_mesh_instances = terrain_mesh_instances.into_inner();
        let mesh_allocator = mesh_allocator.into_inner();

        let Some(RenderMesh2dInstance { mesh_asset_id, .. }) =
            terrain_mesh_instances.get(&item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.get(*mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };
        let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

        let batch_range = item.batch_range();
        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, *index_format);

                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    batch_range.clone(),
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertex_buffer_slice.range, batch_range.clone());
            }
        }
        RenderCommandResult::Success
    }
}

pub struct SetTerrainBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTerrainBindGroup<I> {
    type Param = SRes<Mesh2dBindGroup>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        mesh2d_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        trace!("Setting terrain bind group for phase item");
        let mut dynamic_offsets: [u32; 1] = Default::default();
        let mut offset_count = 0;
        if let Some(dynamic_offset) = item.extra_index().as_dynamic_offset() {
            dynamic_offsets[offset_count] = dynamic_offset.get();
            offset_count += 1;
        }
        pass.set_bind_group(
            I,
            &mesh2d_bind_group.into_inner().value,
            &dynamic_offsets[..offset_count],
        );
        RenderCommandResult::Success
    }
}

pub fn bind_terrain_material(
    mut pipeline: ResMut<TerrainPipeline>,
    render_device: Res<RenderDevice>,
    terrain_materials: Res<Assets<TerrainMaterial>>,
    mut param: StaticSystemParam<<TerrainMaterial as AsBindGroup>::Param>,
) {
    trace!("Binding terrain material");
    // Already prepared
    if pipeline.material_bind_group.is_some() {
        return;
    }

    let Some(terrain_material) = terrain_materials.get(pipeline.terrain_material.id()) else {
        return;
    };

    let Ok(prepared) =
        terrain_material.as_bind_group(&pipeline.material_layout, &render_device, &mut param)
    else {
        return;
    };

    pipeline.material_bind_group = Some(prepared);
}

pub struct SetTerrainMaterialBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTerrainMaterialBindGroup<I> {
    type Param = SRes<TerrainPipeline>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _: &P,
        _view: (),
        _item_query: Option<()>,
        pipeline: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        trace!("Setting terrain material bind group");
        let pipeline = pipeline.into_inner();

        let Some(bind_group) = &pipeline.material_bind_group else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, &bind_group.bind_group, &[]);

        RenderCommandResult::Success
    }
}

pub struct SetTerrainPerChunkBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTerrainPerChunkBindGroup<I> {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = Read<TerrainPerChunkBindGroup>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, Self::ViewQuery>,
        entity: Option<ROQueryItem<'w, Self::ItemQuery>>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        trace!("Setting terrain per-chunk bind group");
        let Some(terrain_per_chunk_bind_group) = entity else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, &terrain_per_chunk_bind_group.value, &[]);

        RenderCommandResult::Success
    }
}

type DrawTerrainMesh2d = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetTerrainBindGroup<1>,
    SetTerrainMaterialBindGroup<2>,
    SetTerrainPerChunkBindGroup<3>,
    DrawTerrain2d,
);
