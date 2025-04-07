use bevy::prelude::*;
use noise_functions::modifiers::{Fbm, Frequency};
use noise_functions::{Constant, Noise, OpenSimplex2, Sample};
use bevy::math::vec2;

use super::{TerrainBase, TerrainSurface, TerrainBaseCell, TerrainChunk};
use super::layers::CellAccessor;

const RANDOM_SEED: u32 = 1;
const SEA_LEVEL: f32 = 64.0f32;
const HEIGHT_LIMIT: f32 = 255.0f32;
const ISLAND_CHUNK_SIZE: u32 = 128;
const ISLAND_RADIUS: f32 = 256.0;

#[derive(Resource)]
pub struct NoiseSource<T> {
    pub noise: T,
}

/// Generates terrain data for a chunk using multiple noise sources
///
/// # Parameters
/// * `chunk_data` - Vector to populate with terrain cell data
/// * `noise_source` - Primary noise source for height generation
pub fn generate_tiles(
    id: IVec2,
    chunk_data: &mut TerrainChunk,
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

    let global_x = id.x * ISLAND_CHUNK_SIZE as i32;
    let global_y = id.y * ISLAND_CHUNK_SIZE as i32;

    chunk_data.parallel_access(IVec2::ZERO, IVec2::new(ISLAND_CHUNK_SIZE as i32, ISLAND_CHUNK_SIZE as i32), 
    |start_pos, end_pos_exclusive, stride, slice| {
        for y_index in 0..end_pos_exclusive.y - start_pos.y {
            let start_index = y_index * (stride as i32);
            for x_index in 0..end_pos_exclusive.x - start_pos.x {
                let item = &mut slice[(start_index + x_index) as usize];
                let x = global_x + start_pos.x + x_index;
                let y = global_y + start_pos.y + y_index;

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
        }
    }).unwrap();
}

pub struct SpawnTilemapCommand {
    pub pos: IVec2
}

impl Command for SpawnTilemapCommand {
    fn apply(self, world: &mut World) {
        let chunk_data = vec![TerrainBaseCell::default(); (ISLAND_CHUNK_SIZE * ISLAND_CHUNK_SIZE) as usize];
        let mut chunk = TerrainChunk { pos: self.pos, data: chunk_data, };
        let noise_source = world.resource();
        generate_tiles(self.pos, &mut chunk, noise_source);
        world.spawn(chunk);
    }
}