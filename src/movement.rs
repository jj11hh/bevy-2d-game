use std::ops::DerefMut;
use bevy::input::ButtonInput;
use bevy::math::Vec3;
use bevy::prelude::{Camera, KeyCode, Projection, Query, Res, Time, Transform, With};
use crate::terrain::TILE_PIXEL_SIZE;

pub fn movement(
    time: Res<Time>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Transform, &mut Projection), With<Camera>>,
) {
    for (mut transform, mut proj) in query.iter_mut() {
        let ortho = match proj.deref_mut() {
            Projection::Orthographic(proj) => proj,
            _ => continue,
        };

        let mut direction = Vec3::ZERO;

        if keyboard_input.pressed(KeyCode::KeyH) {
            direction -= Vec3::new(1.0, 0.0, 0.0) / TILE_PIXEL_SIZE;
        }

        if keyboard_input.pressed(KeyCode::KeyL) {
            direction += Vec3::new(1.0, 0.0, 0.0) / TILE_PIXEL_SIZE;
        }

        if keyboard_input.pressed(KeyCode::KeyK) {
            direction += Vec3::new(0.0, 1.0, 0.0) / TILE_PIXEL_SIZE;
        }

        if keyboard_input.pressed(KeyCode::KeyJ) {
            direction -= Vec3::new(0.0, 1.0, 0.0) / TILE_PIXEL_SIZE;
        }

        if keyboard_input.pressed(KeyCode::KeyZ) {
            ortho.scale += 0.1 / TILE_PIXEL_SIZE;
        }

        if keyboard_input.pressed(KeyCode::KeyX) {
            ortho.scale -= 0.1 / TILE_PIXEL_SIZE;
        }

        ortho.scale = ortho.scale.clamp(0.5 / TILE_PIXEL_SIZE, 5.0 / TILE_PIXEL_SIZE);

        let z = transform.translation.z;
        transform.translation += time.delta_secs() * direction * 500.;
        // Important! We need to restore the Z values when moving the camera around.
        // Bevy has a specific camera setup and this can mess with how our layers are shown.
        transform.translation.z = z;
    }
}