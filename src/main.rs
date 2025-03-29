mod render_layers;
mod terrain;
mod movement;
mod creature;
mod terrain_material;

use bevy::prelude::*;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use movement::movement;
use crate::terrain::{Terrain2dPlugin, TILE_PIXEL_SIZE};


fn main() {

    App::new()
        .add_plugins((
            DefaultPlugins.set(AssetPlugin {
                watch_for_changes_override: Some(true),
                ..Default::default()
            }),
            WorldInspectorPlugin::new(),
            Terrain2dPlugin,
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
