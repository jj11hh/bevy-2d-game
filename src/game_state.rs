use bevy::prelude::*;

/// Event triggered when a game state logic update should occur
#[derive(Event)]
pub struct GameUpdateEvent;

/// Game state
#[derive(Resource, Default, Debug)]
pub(crate) struct GameState {
    pub logic_speed: u32, // 0 = paused, N = N ticks for a fixed frame
    pub logic_frame_count: u32
}

impl GameState {
    /// Check if the game is currently paused
    pub fn is_paused(&self) -> bool {
        self.logic_speed == 0
    }
    
    /// Pause the game
    pub fn pause(&mut self) {
        self.logic_speed = 0;
    }
    
    /// Resume the game with the specified logic speed
    pub fn resume(&mut self, speed: u32) {
        self.logic_speed = speed;
    }
    
    /// Set a new logic speed
    pub fn set_speed(&mut self, speed: u32) {
        self.logic_speed = speed;
    }
    
    /// Get the current frame count
    pub fn frame_count(&self) -> u32 {
        self.logic_frame_count
    }
}

/// System to check if enough ticks have passed and send an update event
pub fn update_game_state(
    state: Res<GameState>,
    mut event_writer: EventWriter<GameUpdateEvent>,
) {
    for _ in 0..state.logic_speed {
        event_writer.send(GameUpdateEvent);
    }
}

/// System to increment frame count when game update events are received
pub fn update_frame_count(
    mut state: ResMut<GameState>,
    mut event_reader: EventReader<GameUpdateEvent>,
) {
    // Count the number of events received
    for _ in event_reader.read() {
        state.logic_frame_count = state.logic_frame_count.wrapping_add(1);
    }
}

pub(crate) struct GameStatePlugin;

impl Plugin for GameStatePlugin {
    fn build(&self, app: &mut App) {
        app
            // Register the GameState resource with default values
            .init_resource::<GameState>()
            // Register the GameUpdateEvent
            .add_event::<GameUpdateEvent>()
            // Add the game state systems to the appropriate schedules
            .add_systems(FixedPreUpdate, update_game_state)
            .add_systems(FixedPostUpdate, update_frame_count);
    }
}