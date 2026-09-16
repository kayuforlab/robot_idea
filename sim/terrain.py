"""Procedural voxel ("boxy") terrain generation.

This mirrors the approach used for rough-terrain generation in
./xml/make_xml.py (kept as read-only reference material for this project):
a flat grid of square box columns, each given a random height, so the robot
has to climb over randomized steps. This module is a fresh implementation
for this project and does not read or modify anything under ./xml.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import mujoco


@dataclass
class VoxelTerrainConfig:
    field_size: tuple[float, float] = (4.0, 4.0)
    voxel_size: float = 0.1
    max_height: float = 0.05
    seed: int = 0
    color: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 1.0)


def add_voxel_terrain(
    spec: mujoco.MjSpec,
    config: VoxelTerrainConfig,
    origin: tuple[float, float] = (0.0, 0.0),
) -> mujoco.MjsBody:
    """Add a grid of randomly-heighted box columns to `spec.worldbody`.

    field_size must be an integer multiple of voxel_size along each axis.
    Returns the terrain body that was added.
    """
    field_x, field_y = config.field_size
    voxel = config.voxel_size

    n_x = round(field_x / voxel)
    n_y = round(field_y / voxel)
    if abs(n_x * voxel - field_x) > 1e-9 or abs(n_y * voxel - field_y) > 1e-9:
        raise ValueError(
            f"field_size {config.field_size} must be an integer multiple of voxel_size {voxel}"
        )

    rng = random.Random(config.seed)
    half_voxel = voxel / 2
    x_start = -field_x / 2 + half_voxel
    y_start = -field_y / 2 + half_voxel
    min_half_height = 1e-4  # mujoco boxes cannot have zero size

    body = spec.worldbody.add_body(
        name="voxel_terrain",
        pos=[origin[0], origin[1], 0.0],
    )
    for i in range(n_x):
        for j in range(n_y):
            half_height = max(rng.uniform(0.0, config.max_height) / 2, min_half_height)
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_voxel, half_voxel, half_height],
                pos=[x_start + i * voxel, y_start + j * voxel, half_height],
                group=1,
                friction=[1, 0.1, 0.1],
                rgba=list(config.color),
            )
    return body
