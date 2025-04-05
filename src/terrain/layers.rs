use std::marker::PhantomData;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::utils::HashMap;
use bevy::render::render_resource::{AsBindGroup,ShaderRef};

use super::ISLAND_CHUNK_SIZE;

pub trait TerrainMaterial: AsBindGroup + Clone + Sized + FromWorld + Send + Sync {
    fn fragment_shader() -> ShaderRef { ShaderRef::Default }
}

pub trait TerrainCellData: Sized + Clone + Copy + Send + Sync {}

pub trait AsCellAccessor<CellData: TerrainCellData>: {
    fn get_cell(&self, pos: IVec2) -> Option<CellData>;
    fn set_cell(&mut self, pos: IVec2, data: CellData) -> bool;
    fn modify_cell<F>(&mut self, pos: IVec2, op: F) -> bool where F: Fn(&mut CellData);
}

#[derive(Resource)]
pub struct TerrainTrunkMap<CellData: TerrainCellData> {
    pub map: HashMap<IVec2, Entity>,
    _phantom: PhantomData<CellData>,
}

pub trait TrunkCellAccessor<CellData: TerrainCellData>: AsCellAccessor<CellData> + Component {}

#[derive(SystemParam)]
pub struct CellAccessor<'w, 's, CellData, Trunk>
where 
    CellData: TerrainCellData + 'static,
    Trunk: TrunkCellAccessor<CellData> + 'static,
{
    pub query: Query<'w, 's, &'static mut Trunk>,
    pub trunk_map: ResMut<'w, TerrainTrunkMap<CellData>>,
}

impl<'w, 's, CellData, Trunk> AsCellAccessor<CellData> for CellAccessor<'w, 's, CellData, Trunk>
where
    CellData: TerrainCellData + 'static,
    Trunk: TrunkCellAccessor<CellData> + 'static,
{
    fn get_cell(&self, pos: IVec2) -> Option<CellData> {
        let chunk_size = ISLAND_CHUNK_SIZE as i32;
        let chunk_pos = IVec2::new(
            pos.x.div_euclid(chunk_size),
            pos.y.div_euclid(chunk_size),
        );
        let local_pos = IVec2::new(
            pos.x.rem_euclid(chunk_size),
            pos.y.rem_euclid(chunk_size),
        );
        
        if let Some(entity) = self.trunk_map.map.get(&chunk_pos) {
            if let Ok(trunk) = self.query.get(*entity) {
                trunk.get_cell(local_pos)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn set_cell(&mut self, pos: IVec2, data: CellData) -> bool {
        let chunk_size = ISLAND_CHUNK_SIZE as i32;
        let chunk_pos = IVec2::new(
            pos.x.div_euclid(chunk_size),
            pos.y.div_euclid(chunk_size),
        );
        let local_pos = IVec2::new(
            pos.x.rem_euclid(chunk_size),
            pos.y.rem_euclid(chunk_size),
        );
        
        if let Some(entity) = self.trunk_map.map.get(&chunk_pos) {
            if let Ok(mut trunk) = self.query.get_mut(*entity) {
                trunk.set_cell(local_pos, data)
            } else {
                false
            }
        } else {
            false
        }
    }

    fn modify_cell<F>(&mut self, pos: IVec2, op: F) -> bool 
    where 
        F: Fn(&mut CellData) 
    {
        let chunk_size = ISLAND_CHUNK_SIZE as i32;
        let chunk_pos = IVec2::new(
            pos.x.div_euclid(chunk_size),
            pos.y.div_euclid(chunk_size),
        );
        let local_pos = IVec2::new(
            pos.x.rem_euclid(chunk_size),
            pos.y.rem_euclid(chunk_size),
        );
        
        if let Some(entity) = self.trunk_map.map.get(&chunk_pos) {
            if let Ok(mut trunk) = self.query.get_mut(*entity) {
                trunk.modify_cell(local_pos, op)
            } else {
                false
            }
        } else {
            false
        }
    }
}