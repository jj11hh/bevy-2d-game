use std::marker::PhantomData;
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::lifetimeless::{Read, SRes};
use bevy::ecs::system::SystemParamItem;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_phase::{
    PhaseItem, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass,
};
use bevy::sprite::{
    Mesh2dBindGroup, Mesh2dPipeline, RenderMesh2dInstance, SetMesh2dViewBindGroup,
};

use bevy::render::sync_world::MainEntityHashMap;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::*;
use bevy::render::texture::GpuImage;
use bevy::prelude::*;

use super::layers::TerrainMaterial;

#[derive(Resource, Deref, DerefMut, Default)]
pub struct RenderTerrainMeshInstances(MainEntityHashMap<RenderMesh2dInstance>);

#[derive(Component, Debug, Clone)]
pub struct TerrainPerChunkBindGroup {
    pub value: BindGroup,
}

#[derive(Resource)]
pub struct TerrainPipeline<M: TerrainMaterial> {
    /// This pipeline wraps the standard [`Mesh2dPipeline`]
    pub mesh2d_pipeline: Mesh2dPipeline,
    pub material_layout: BindGroupLayout,
    pub per_chunk_data_layout: BindGroupLayout,
    pub material_bind_group: Option<PreparedBindGroup<<M as AsBindGroup>::Data>>,
    pub dummy_black_gpu_image: GpuImage,
    pub terrain_shader: Handle<Shader>,
    pub terrain_material: M,
}

pub struct TerrainPerChunkData {
    pub terrain_map_gpu_image: GpuImage,
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
        trace!(
            "Rendering DrawTerrain2d for entity: {:?}",
            item.main_entity()
        );
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

pub struct SetTerrainMaterialBindGroup<const I: usize, M: TerrainMaterial + 'static> {
    marker: PhantomData<M>,
}

impl<P: PhaseItem, const I: usize, M: TerrainMaterial + 'static> RenderCommand<P>
    for SetTerrainMaterialBindGroup<I, M>
{
    type Param = SRes<TerrainPipeline<M>>;
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

pub type DrawTerrainMesh2d<M> = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetTerrainBindGroup<1>,
    SetTerrainMaterialBindGroup<2, M>,
    SetTerrainPerChunkBindGroup<3>,
    DrawTerrain2d,
);
