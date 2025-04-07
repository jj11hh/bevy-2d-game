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

#[derive(Debug)]
pub enum CellAccessError {
    ChunkNotFound,
    ChunkAccessFailed,
}

pub trait CellAccessor<CellData: TerrainCellData>: {
    fn get_cell(&self, pos: IVec2) -> Option<CellData>;
    fn set_cell(&mut self, pos: IVec2, data: CellData) -> Result<(), CellAccessError>;
    fn modify_cell<F>(&mut self, pos: IVec2, op: F) -> Result<(), CellAccessError> where F: Fn(&mut CellData);
    fn bulk_access<F>(&mut self, start_pos: IVec2, end_pos_exclusive: IVec2, op: F) -> Result<(), CellAccessError> 
    where F: Fn(IVec2, IVec2, usize, &mut [CellData]);
    fn parallel_access<F>(&mut self, start_pos: IVec2, end_pos_exclusive: IVec2, op: F) -> Result<(), CellAccessError> 
    where F: Fn(IVec2, IVec2, usize, &mut [CellData]) + Send + Sync;
}

#[derive(Resource, Default)]
pub struct TerrainChunkMap<CellData: TerrainCellData> {
    pub map: HashMap<IVec2, Entity>,
    _phantom: PhantomData<CellData>,
}

pub trait ChunkCellAccessor<CellData: TerrainCellData>: CellAccessor<CellData> + Component {}

#[derive(SystemParam)]
pub struct GlobalCellAccessor<'w, 's, CellData, Chunk>
where 
    CellData: TerrainCellData + 'static,
    Chunk: ChunkCellAccessor<CellData> + 'static,
{
    pub query: Query<'w, 's, &'static mut Chunk>,
    pub chunk_map: ResMut<'w, TerrainChunkMap<CellData>>,
}

impl<'w, 's, CellData, Chunk> GlobalCellAccessor<'w, 's, CellData, Chunk>
where
    CellData: TerrainCellData + 'static,
    Chunk: ChunkCellAccessor<CellData> + 'static,
{
    pub fn process_chunk_range<C>(
        &mut self,
        start_pos: IVec2,
        end_pos_exclusive: IVec2,
        chunk_access: C,
    ) -> Result<(), CellAccessError>
    where
        C: Fn(&mut Chunk, IVec2, IVec2, IVec2) -> Result<(), CellAccessError>,
    {
        let chunk_size = ISLAND_CHUNK_SIZE as i32;
        let start_chunk = IVec2::new(
            start_pos.x.div_euclid(chunk_size),
            start_pos.y.div_euclid(chunk_size),
        );
        let end_chunk = IVec2::new(
            end_pos_exclusive.x.div_euclid(chunk_size),
            end_pos_exclusive.y.div_euclid(chunk_size),
        );

        let mut any_success = false;
        let mut any_failure = false;
        
        for chunk_y in start_chunk.y..=end_chunk.y {
            for chunk_x in start_chunk.x..=end_chunk.x {
                let chunk_pos = IVec2::new(chunk_x, chunk_y);
                
                let chunk_start = chunk_pos * chunk_size;
                let chunk_end = chunk_start + IVec2::splat(chunk_size - 1);
                
                let local_start = IVec2::new(
                    start_pos.x.max(chunk_start.x) - chunk_start.x,
                    start_pos.y.max(chunk_start.y) - chunk_start.y,
                );
                let local_end = IVec2::new(
                    end_pos_exclusive.x.min(chunk_end.x) - chunk_start.x,
                    end_pos_exclusive.y.min(chunk_end.y) - chunk_start.y,
                );

                if let Some(entity) = self.chunk_map.map.get(&chunk_pos) {
                    if let Ok(mut chunk) = self.query.get_mut(*entity) {
                        let chunk_base = chunk_pos * chunk_size;
                        if chunk_access(&mut chunk, chunk_base, local_start, local_end).is_ok() {
                            any_success = true;
                        } else {
                            any_failure = true;
                        }
                    } else {
                        any_failure = true;
                    }
                } else {
                    any_failure = true;
                }
            }
        }

        if any_failure {
            if any_success {
                Err(CellAccessError::ChunkAccessFailed)
            } else {
                Err(CellAccessError::ChunkNotFound)
            }
        } else {
            Ok(())
        }
    }
}

impl<'w, 's, CellData, Chunk> CellAccessor<CellData> for GlobalCellAccessor<'w, 's, CellData, Chunk>
where
    CellData: TerrainCellData + 'static,
    Chunk: ChunkCellAccessor<CellData> + 'static,
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
        
        if let Some(entity) = self.chunk_map.map.get(&chunk_pos) {
            if let Ok(chunk) = self.query.get(*entity) {
                chunk.get_cell(local_pos)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn set_cell(&mut self, pos: IVec2, data: CellData) -> Result<(), CellAccessError> {
        let chunk_size = ISLAND_CHUNK_SIZE as i32;
        let chunk_pos = IVec2::new(
            pos.x.div_euclid(chunk_size),
            pos.y.div_euclid(chunk_size),
        );
        let local_pos = IVec2::new(
            pos.x.rem_euclid(chunk_size),
            pos.y.rem_euclid(chunk_size),
        );
        
        if let Some(entity) = self.chunk_map.map.get(&chunk_pos) {
            if let Ok(mut chunk) = self.query.get_mut(*entity) {
                chunk.set_cell(local_pos, data)
            } else {
                Err(CellAccessError::ChunkAccessFailed)
            }
        } else {
            Err(CellAccessError::ChunkNotFound)
        }
    }

    fn modify_cell<F>(&mut self, pos: IVec2, op: F) -> Result<(), CellAccessError>
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
        
        if let Some(entity) = self.chunk_map.map.get(&chunk_pos) {
            if let Ok(mut chunk) = self.query.get_mut(*entity) {
                chunk.modify_cell(local_pos, op)
            } else {
                Err(CellAccessError::ChunkAccessFailed)
            }
        } else {
            Err(CellAccessError::ChunkNotFound)
        }
    }
   
    fn bulk_access<F>(&mut self, start_pos: IVec2, end_pos_exclusive: IVec2, op: F) -> Result<(), CellAccessError>
    where F: Fn(IVec2, IVec2, usize, &mut [CellData])
    {
        let op = &op;
        self.process_chunk_range(start_pos, end_pos_exclusive, |chunk, chunk_base, start, end| {
            chunk.bulk_access(start, end, |start_pos, end_pos, stride, cells| {
                op(start_pos + chunk_base, end_pos + chunk_base, stride, cells)
            })
        })
    }

    fn parallel_access<F>(&mut self, start_pos: IVec2, end_pos_exclusive: IVec2, op: F) -> Result<(), CellAccessError>
    where F: Fn(IVec2, IVec2, usize, &mut [CellData]) + Send + Sync
    {
        let op = &op;
        self.process_chunk_range(start_pos, end_pos_exclusive, |chunk, chunk_base, start, end| {
            chunk.parallel_access(start, end, |start_pos, end_pos, stride, cells| {
                op(start_pos + chunk_base, end_pos + chunk_base, stride, cells)
            })
        })
    }
}