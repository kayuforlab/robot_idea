from sim.simulate import run
from sim.terrain import SandTerrainConfig, SlopeTerrainConfig, VoxelTerrainConfig
from sim.world import build_model




# ROBOT_NAME = "2wheel_rover/normal"
# ROBOT_NAME = "2wheel_rover/one_way_grouser"
# ROBOT_NAME = "2wheel_rover/fan"

#armがある2輪
# ROBOT_NAME = "2wheel_arm/rotate"
# ROBOT_NAME = "2wheel_arm/wheel_slider"
# ROBOT_NAME = "2wheel_arm/body_slider"
ROBOT_NAME = "2wheel_arm/body_slider_rotate"


#砂地形
# resolution: heightfieldの格子間隔（細かさ）。見た目の滑らかさだけに影響し、沈み込みには無関係。
# dune_height: 表面のリップル（起伏）の見た目の高さ。これも沈み込みには無関係。
# firmness: 砂の締まり具合（0=緩い砂、1=固く締まった砂）という物性値であって、
#   「何cm沈む」という指定ではない。実際に何cm沈むかはロボットの重さとの兼ね合いで
#   物理演算が結果として決める（ロボットは常に地形の上空から自由落下して着地する）。
# TERRAIN_CONFIG = SandTerrainConfig(
#     field_size=(4.0, 4.0),
#     resolution=0.05,
#     dune_height=0.04,
#     firmness=0.08,
#     seed=0,
# )

#傾斜
# TERRAIN_CONFIG = SlopeTerrainConfig(
#     field_size=(6.0, 6.0),
#     incline=0.2,
# )

#ボクセル
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
