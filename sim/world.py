"""Assemble a full MuJoCo model: base scene + terrain + one robot.

Robots are stored per the project convention ./<robot_name>/xml/robot.xml.
This module loads that file as its own MjSpec and attaches it onto the
base world scene (sim/xml/world.xml) at a chosen spawn frame, after adding
procedurally generated terrain (see sim/terrain.py for the available
terrain types).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from sim.terrain import TerrainConfig, VoxelTerrainConfig, add_terrain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORLD_XML = Path(__file__).resolve().parent / "xml" / "world.xml"


def robot_xml_path(robot_name: str) -> Path:
    return PROJECT_ROOT / robot_name / "xml" / "robot.xml"


def build_model(
    robot_name: str,
    terrain_config: TerrainConfig | None = None,
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.3),
) -> mujoco.MjModel:
    """Build the combined MjModel for `robot_name` on generated terrain."""
    if terrain_config is None:
        terrain_config = VoxelTerrainConfig()

    world_spec = mujoco.MjSpec.from_file(str(WORLD_XML))
    add_terrain(world_spec, terrain_config)

    robot_path = robot_xml_path(robot_name)
    if not robot_path.exists():
        raise FileNotFoundError(f"no robot xml found at {robot_path}")
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))

    spawn_frame = world_spec.worldbody.add_frame(pos=list(spawn_pos))
    world_spec.attach(robot_spec, prefix="", frame=spawn_frame)

    return world_spec.compile()
