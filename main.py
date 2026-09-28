from sim.simulate import run
from sim.terrain import CaveTerrainConfig, SandTerrainConfig, SlopeTerrainConfig, VoxelTerrainConfig
from sim.world import build_model




# ROBOT_NAME = "2wheel_rover/normal"
# ROBOT_NAME = "2wheel_rover/one_way_grouser"
# ROBOT_NAME = "2wheel_rover/fan"
ROBOT_NAME = "2wheel_rover/tripod"
# ROBOT_NAME = "2wheel_rover/tripod_eccentric"

#armがある2輪
# ROBOT_NAME = "2wheel_arm/rotate"
# ROBOT_NAME = "2wheel_arm/wheel_slider"
# ROBOT_NAME = "2wheel_arm/body_slider"
# ROBOT_NAME = "2wheel_arm/body_slider_rotate"

#bodyが中央で2分割され中央の1モータで円柱軸まわりに回転、左右の端に伸縮スライダーが付く(アクチュエータ計3)
ROBOT_NAME = "slider/slider_rotate"



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

#洞窟
# undulation_height: 床自体の緩やかな起伏の高さ
# rock_spacing: 岩を並べる格子の間隔。岩サイズに対して小さいほど隙間なく密に敷き詰められる
# jitter: 格子から各岩をずらす量
# min_rock_size / max_rock_size: 岩サイズの範囲
# round_ratio: 丸い岩（球・楕円体）の存在比率。
# TERRAIN_CONFIG = CaveTerrainConfig(
#     field_size=(4.0, 4.0),
#     undulation_height=0.05,
#     rock_spacing=0.1,
#     jitter=0.6,
#     min_rock_size=0.035,
#     max_rock_size=0.2,
#     round_ratio=1,
#     seed=0,
# )


def main():
    model = build_model(
        robot_name=ROBOT_NAME,
        terrain_config=TERRAIN_CONFIG,
        spawn_pos=(0.0, 0.0, TERRAIN_CONFIG.max_height + 0.3),
    )
    run(model)


if __name__ == "__main__":
    main()
