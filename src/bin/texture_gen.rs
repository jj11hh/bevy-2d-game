use bevy::math::Vec2;

struct PermutationTable {
    data: Vec<Vec2>,
    size: usize,
}

impl PermutationTable {
    pub fn new(size: usize) -> Self {
        Self{
            data: vec![Vec2::default(); size * size],
            size
        }
    }
    
    pub fn randomize(&mut self, seed: u32) {
        todo!()
    }
}

pub trait TileableNoise {
    fn noise(permutation_table: &PermutationTable, position: Vec2) -> f32;
} 

struct TileablePerlinNoise;

impl TileableNoise for TileablePerlinNoise {
    fn noise(permutation_table: &PermutationTable, position: Vec2) -> f32 {
        let left = position.x.floor().rem_euclid(permutation_table.size as f32) as usize;
        let right = position.x.ceil().rem_euclid(permutation_table.size as f32) as usize;
        let top = position.y.ceil().rem_euclid(permutation_table.size as f32) as usize;
        let bottom = position.y.floor().rem_euclid(permutation_table.size as f32) as usize;
        todo!()
    }
}


fn main() {
    println!("Hello, World!");
}