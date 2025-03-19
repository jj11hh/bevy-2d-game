#import bevy_sprite::mesh2d_vertex_output::VertexOutput

const SEA_LEVEL: f32 = 64.0;
const HEIGHT_LIMIT: f32 = 255.0;
const ISLAND_CHUNK_SIZE: u32 = 128;
const TILE_PIXEL_SIZE: f32 = 32.0;

const LIGHT_OCEAN_COLOR: vec3<f32> = vec3<f32>(0.18, 0.835, 1);
const DARK_OCEAN_COLOR: vec3<f32> = vec3<f32>(0, 0.306, 0.38);

const DEFAULT_ROCK_COLOR: vec3<f32> = vec3<f32>(0.3, 0.22, 0.2);

const BASE_ROCK: u32 = 0;
const BASE_SAND: u32 = 1;
const BASE_SOIL: u32 = 2;
const BASE_MUD:  u32 = 3;

@group(2) @binding(0) var<uniform> render_mode: u32;

@group(2) @binding(1) var super_perlin_texture: texture_2d<f32>;
@group(2) @binding(2) var super_perlin_sampler: sampler;

@group(2) @binding(3) var grainy_texture: texture_2d<f32>;
@group(2) @binding(4) var grainy_sampler: sampler;

@group(3) @binding(0) var terrain_map_texture: texture_2d<f32>;
@group(3) @binding(1) var terrain_map_sampler: sampler;

fn rock(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    let noise = textureSample(super_perlin_texture, super_perlin_sampler, mesh.world_position.xy * 0.25);
    let base_color = DEFAULT_ROCK_COLOR;
    let color = mix(base_color * 0.80, base_color, noise.r);

    return vec4<f32>(color, 1.0);
}

fn shade_tile(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    // let height = tile_data.x;
    // return vec4<f32>(height, height, height, 1.0);
    return rock(mesh, tile_data);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let load_pos_f = mesh.uv * f32(ISLAND_CHUNK_SIZE);

    let load_pos_i = vec2<i32>(load_pos_f);
    let sample_dir = load_pos_f - vec2<f32>(load_pos_i) - vec2<f32>(0.5, 0.5);
    let sample_sign = vec4<i32>(vec2<i32>(sign(sample_dir)), 0, 0);

    let mask0 = textureLoad(terrain_map_texture, load_pos_i + vec2<i32>(sample_sign.z, 0), 0);
    let mask1 = textureLoad(terrain_map_texture, load_pos_i + vec2<i32>(sample_sign.x, sample_sign.z), 0);
    let mask2 = textureLoad(terrain_map_texture, load_pos_i + vec2<i32>(sample_sign.z, sample_sign.y), 0);
    let mask3 = textureLoad(terrain_map_texture, load_pos_i + vec2<i32>(sample_sign.x, sample_sign.y), 0);

    var color: vec4<f32> = shade_tile(mesh, mask0);

    var uv_in_tile: vec2<f32> = fract(mesh.world_position.xy);
    let uv_overlay = vec4<f32>(uv_in_tile, 1.0, 1.0); // Debug UV

    if render_mode != 0 {
        return color + uv_overlay * 0.1;
    }

    return color;
}