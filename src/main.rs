mod movement;
mod creature;
mod game_state;
mod terrain;

use bevy::prelude::*;
use bevy_inspector_egui::bevy_egui::EguiPlugin;
use bevy_inspector_egui::quick::{ResourceInspectorPlugin, WorldInspectorPlugin};
use movement::movement;
use crate::terrain::{Terrain2dPlugin, TerrainChunkRenderers, TILE_PIXEL_SIZE};
use crate::game_state::GameStatePlugin;


fn main() {

    App::new()
        .add_plugins((
            DefaultPlugins.set(AssetPlugin {
            watch_for_changes_override: Some(true),
            ..Default::default()
        }),
        Terrain2dPlugin,
        GameStatePlugin,
        EguiPlugin { enable_multipass_for_primary_context: false },
        ResourceInspectorPlugin::<TerrainChunkRenderers>::default(),
        WorldInspectorPlugin::new(),
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
    
    let projection = Projection::Orthographic(ortho);

    commands.spawn((
        Camera2d,
        projection
    ));
}
