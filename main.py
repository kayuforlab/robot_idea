from sim.simulate import run
from sim.terrain import VoxelTerrainConfig
from sim.world import build_model




ROBOT_NAME = "2wheel_rover/normal"
ROBOT_NAME = "2wheel_rover/one_way_grouser"
ROBOT_NAME = "2wheel_rover/fan"


TERRAIN_CONFIG = VoxelTerrainConfig(
    field_size=(4.0, 4.0),
    voxel_size=0.1,
    max_height=0.05,
    seed=0,
)


def main():
    model = build_model(
        robot_name=ROBOT_NAME,
        terrain_config=TERRAIN_CONFIG,
        spawn_pos=(0.0, 0.0, TERRAIN_CONFIG.max_height + 0.3),
    )
    run(model)


if __name__ == "__main__":
    main()
