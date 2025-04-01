#import bevy_sprite::mesh2d_vertex_output::VertexOutput

const SEA_LEVEL: f32 = 64.0;
const HEIGHT_LIMIT: f32 = 255.0;
const ISLAND_CHUNK_SIZE: u32 = 64;
const TILE_PIXEL_SIZE: f32 = 32.0;

const LIGHT_OCEAN_COLOR: vec3<f32> = vec3<f32>(0.18, 0.835, 1);
const DARK_OCEAN_COLOR: vec3<f32> = vec3<f32>(0, 0.306, 0.38);

const DEFAULT_ROCK_COLOR: vec3<f32> = vec3<f32>(0.3, 0.22, 0.2);
const DEFAULT_SAND_COLOR: vec3<f32> = vec3<f32>(0.95, 0.88, 0.6);
const DEFAULT_SOIL_COLOR: vec3<f32> = vec3<f32>(0.5, 0.35, 0.2);
const DEFAULT_MUD_COLOR: vec3<f32> = vec3<f32>(0.4, 0.3, 0.15);

const WATER_COLOR: vec3<f32> = vec3<f32>(0.12, 0.56, 0.85);
const GRASS_COLOR: vec3<f32> = vec3<f32>(0.15, 0.45, 0.15);
const SNOW_COLOR: vec3<f32> = vec3<f32>(0.95, 0.95, 0.98);

const BASE_ROCK: u32 = 0;
const BASE_SAND: u32 = 1;
const BASE_SOIL: u32 = 2;
const BASE_MUD:  u32 = 3;

const SURFACE_BARE: u32 = 0;
const SURFACE_WATER: u32 = 1;
const SURFACE_GRASS: u32 = 2;
const SURFACE_SNOW: u32 = 3;

struct TerrainMaterialUniform {
    mode: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
    time: vec4<f32>, // elapsed secs, delta_secs, sin(elapsed secs), cos(elapsed secs)
}

@group(2) @binding(0) var<uniform> terrainMaterial: TerrainMaterialUniform;

@group(2) @binding(1) var super_perlin_texture: texture_2d<f32>;
@group(2) @binding(2) var super_perlin_sampler: sampler;

@group(2) @binding(3) var grainy_texture: texture_2d<f32>;
@group(2) @binding(4) var grainy_sampler: sampler;

@group(3) @binding(0) var terrain_map_texture_rt: texture_2d<f32>;
@group(3) @binding(1) var terrain_map_texture_lt: texture_2d<f32>;
@group(3) @binding(2) var terrain_map_texture_rb: texture_2d<f32>;
@group(3) @binding(3) var terrain_map_texture_lb: texture_2d<f32>;
@group(3) @binding(4) var terrain_map_sampler: sampler;

fn rock(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    let noise = textureSample(super_perlin_texture, super_perlin_sampler, mesh.world_position.xy * 0.25);
    let base_color = DEFAULT_ROCK_COLOR;
    let color = mix(base_color * 0.80, base_color, noise.r);
    return vec4<f32>(color, 1.0);
}

fn sand(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    // Basic grainy texture
    let noise = textureSample(grainy_texture, texture_sampler, mesh.world_position.xy * 0.3);
    
    // Add wind-blown patterns using directional noise
    let wind_angle = 0.7; // Wind direction angle
    let wind_dir = vec2<f32>(cos(wind_angle), sin(wind_angle));
    let wind_noise = textureSample(
        super_perlin_texture,
        texture_sampler,
        mesh.world_position.xy * 0.15 + wind_dir * textureSample(grainy_texture, texture_sampler, mesh.world_position.xy * 0.05).r * 0.2
    );
    
    let base_color = DEFAULT_SAND_COLOR;
    // Mix the base color with both the grainy texture and the wind pattern
    let color = mix(base_color * 0.6, base_color, noise.r);
    let wind_color = mix(color * 0.9, color * 1.1, wind_noise.r);
    
    return vec4<f32>(wind_color, 1.0);
}

fn soil(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    let noise = textureSample(super_perlin_texture, texture_sampler, mesh.world_position.xy * 0.35);
    let base_color = DEFAULT_SOIL_COLOR;
    let color = mix(base_color * 0.85, base_color * 1.05, noise.r);
    return vec4<f32>(color, 1.0);
}

fn mud(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    let noise = textureSample(super_perlin_texture, texture_sampler, mesh.world_position.xy * 0.4);
    let noise2 = textureSample(grainy_texture, texture_sampler, mesh.world_position.xy * 0.25);
    let base_color = DEFAULT_MUD_COLOR;
    let color = mix(base_color * 0.9, base_color * 1.05, noise.r * noise2.r);
    return vec4<f32>(color, 1.0);
}

fn apply_water_surface(base_color: vec4<f32>, mesh: VertexOutput, height: f32) -> vec4<f32> {
    let water_noise = textureSample(super_perlin_texture, texture_sampler,
        mesh.world_position.xy * 0.1 + vec2<f32>(sin(mesh.world_position.x * 0.05), cos(mesh.world_position.y * 0.05)));
    
    let water_depth = max(0.0, (SEA_LEVEL - height) / SEA_LEVEL);
    let water_color = mix(LIGHT_OCEAN_COLOR, DARK_OCEAN_COLOR, water_depth);
    let wave = sin(mesh.world_position.x * 0.1) * cos(mesh.world_position.y * 0.1) * 0.5 + 0.5;
    let final_color = mix(base_color.rgb, water_color, min(0.7 + water_depth * 0.3, 0.95));
    let highlight = water_noise.r * wave * 0.15;
    return vec4<f32>(final_color + highlight, 1.0);
}

fn apply_grass_surface(base_color: vec4<f32>, mesh: VertexOutput, height: f32) -> vec4<f32> {
    let noise = textureSample(grainy_texture, texture_sampler, mesh.world_position.xy * 0.5);
    let grass_factor = clamp((height - SEA_LEVEL) / (HEIGHT_LIMIT * 0.7), 0.8, 1.0);
    let grass_variation = mix(GRASS_COLOR * 0.85, GRASS_COLOR * 1.1, noise.r);
    let final_color = mix(base_color.rgb, grass_variation, grass_factor);
    return vec4<f32>(final_color, 1.0);
}

fn apply_snow_surface(base_color: vec4<f32>, mesh: VertexOutput, height: f32) -> vec4<f32> {
    let noise = textureSample(super_perlin_texture, texture_sampler, mesh.world_position.xy * 0.4);
    let snow_factor = 1.0; //smoothstep(HEIGHT_LIMIT * 0.7, HEIGHT_LIMIT * 0.85, height);
    let snow_variation = mix(SNOW_COLOR * 0.95, SNOW_COLOR, noise.r);
    let final_color = mix(base_color.rgb, snow_variation, snow_factor);
    return vec4<f32>(final_color, 1.0);
}

fn get_base_color(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    let base_type = u32(tile_data.y * 255.0);
    switch base_type {
        case BASE_SAND: { return sand(mesh, tile_data); }
        case BASE_SOIL: { return soil(mesh, tile_data); }
        case BASE_MUD: { return mud(mesh, tile_data); }
        default: { return rock(mesh, tile_data); }
    }
}

fn shade_tile(mesh: VertexOutput, tile_data: vec4<f32>) -> vec4<f32> {
    let height = tile_data.x * 255.0;
    let surface_type = u32(tile_data.z * 255.0);
    let base_color = get_base_color(mesh, tile_data);
    
    switch surface_type {
        case SURFACE_WATER: { return apply_water_surface(base_color, mesh, height); }
        case SURFACE_GRASS: { return apply_grass_surface(base_color, mesh, height); }
        case SURFACE_SNOW: { return apply_snow_surface(base_color, mesh, height); }
        default: { return base_color; }
    }
}

fn shaperstep(x: f32, softness: f32) -> f32 {
    let clamped_x = clamp(x, 0.0, 1.0);
    let is_less_than_half = f32(clamped_x < 0.5);
    
    // Calculate both branches
    let lower_half = pow(2.0 * clamped_x, softness) / 2.0;
    let upper_half = 1.0 - pow(2.0 - 2.0 * clamped_x, softness) / 2.0;
    
    // Select the appropriate result based on x value
    return lower_half * is_less_than_half + upper_half * (1.0 - is_less_than_half);
}

fn load_terrain_data(load_pos: vec2<i32>) -> vec4<f32> {
    let half_size = i32(ISLAND_CHUNK_SIZE/2);
    let size = i32(ISLAND_CHUNK_SIZE);
    let pos = load_pos - vec2<i32>(half_size, half_size);
    if pos.x >= 0 {
        if pos.y >= 0 {
            return textureLoad(terrain_map_texture_rb, pos, 0);
        }
        else {
            return textureLoad(terrain_map_texture_rt, vec2<i32>(pos.x, size + pos.y), 0);
        }
    }
    else {
        if pos.y >= 0 {
            return textureLoad(terrain_map_texture_lb, vec2<i32>(size + pos.x, pos.y), 0);
        }
        else {
            return textureLoad(terrain_map_texture_lt, vec2<i32>(size + pos.x, size + pos.y), 0);
        }
    }
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let load_pos_f = mesh.uv * f32(ISLAND_CHUNK_SIZE);
    let load_pos_i = vec2<i32>(load_pos_f);
    let sample_dir = load_pos_f - vec2<f32>(load_pos_i) - vec2<f32>(0.5, 0.5);
    let sample_sign = vec4<i32>(vec2<i32>(sign(sample_dir)), 0, 0);

    let data0 = load_terrain_data(load_pos_i + vec2<i32>(sample_sign.z, 0));
    let data1 = load_terrain_data(load_pos_i + vec2<i32>(sample_sign.x, sample_sign.z));
    let data2 = load_terrain_data(load_pos_i + vec2<i32>(sample_sign.z, sample_sign.y));
    let data3 = load_terrain_data(load_pos_i + vec2<i32>(sample_sign.x, sample_sign.y));

    // Get colors for all four data texture
    let color0 = shade_tile(mesh, data0);
    let color1 = shade_tile(mesh, data1);
    let color2 = shade_tile(mesh, data2);
    let color3 = shade_tile(mesh, data3);

    // Calculate sharper interpolation factors based on sample_dir
    // For values between 0.1-0.9, use the current tile's color completely
    let abs_sample_dir = abs(sample_dir);
    
    // Create interpolation factors that are 0 in the middle range and only interpolate at edges
    let factor_x = shaperstep(abs_sample_dir.x, 5.0);
    let factor_y = shaperstep(abs_sample_dir.y, 5.0);
    
    // Blend colors using the sharper interpolation
    // First mix horizontally: color0 with color2, and color1 with color3
    let color_h1 = mix(color0, color2, factor_y);
    let color_h2 = mix(color1, color3, factor_y);
    
    // Then mix vertically
    var color = mix(color_h1, color_h2, factor_x);
    
    var uv_in_tile: vec2<f32> = fract(mesh.world_position.xy);

    let uv_overlay = vec4<f32>(uv_in_tile, 1.0, 1.0); // Debug UV

    if terrainMaterial.mode != 0 {
        return color + uv_overlay * 0.1;
    }

    return color;
}
