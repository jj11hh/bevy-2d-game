mod render_layers;
mod terrain;
mod movement;
mod creature;
mod game_state;

use bevy::prelude::*;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use movement::movement;
use crate::terrain::{Terrain2dPlugin, TILE_PIXEL_SIZE};
use crate::game_state::GameStatePlugin;

const RANDOM_SEED: u32 = 1;

const SEA_LEVEL: f32 = 64.0f32;
const HEIGHT_LIMIT: f32 = 255.0f32;

#[derive(Resource)]
struct NoiseSource<T> {
    noise: T
}

fn main() {

    App::new()
        .add_plugins((
            DefaultPlugins.set(AssetPlugin {
                watch_for_changes_override: Some(true),
                ..Default::default()
            }),
            WorldInspectorPlugin::new(),
            Terrain2dPlugin,
            GameStatePlugin,
        ))
        .add_systems(Startup, startup)
        .add_systems(Update, movement)
        .run();
}

fn startup(
    mut commands: Commands
) {
    let mut ortho = OrthographicProjection::default_2d();
    ortho.scale = 1.0 / TILE_PIXEL_SIZE;

    commands.spawn((
        Camera2d,
        ortho
    ));
}
