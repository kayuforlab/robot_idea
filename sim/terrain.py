"""Procedural terrain generation: rough ("boxy") ground, a sandy field, and a
uniform slope.

The rough-terrain approach mirrors ./xml/make_xml.py (kept as read-only
reference material for this project): a flat grid of square box columns,
each given a random height, so the robot has to climb over randomized
steps. This module is a fresh implementation for this project and does not
read or modify anything under ./xml.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class VoxelTerrainConfig:
    """Randomized rocky terrain: a grid of box columns with random heights."""

    field_size: tuple[float, float] = (4.0, 4.0)
    voxel_size: float = 0.1
    max_height: float = 0.05
    seed: int = 0
    color: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 1.0)
    friction: tuple[float, float, float] = (1.0, 0.1, 0.1)


#: Fixed physical headroom (m) the sand surface is raised above the world's
#: shared invisible floor (sim/xml/world.xml) so there's real room for a
#: wheel to physically sink in. This is *not* "how deep it sinks" -- it's
#: just enough slack for the contact physics to operate in; how far
#: anything actually settles into that slack is decided by mass vs.
#: firmness, at simulation time, like real ground.
_SAND_HEADROOM = 0.2


@dataclass
class SandTerrainConfig:
    """Sandy terrain: a smooth, continuous heightfield of fine ripples (not
    a grid of discrete boxes like VoxelTerrainConfig -- that just reads as a
    finer version of the same rocky terrain), soft enough that a robot
    dropped onto it visibly sinks in and settles under its own weight.

    `dune_height` (the visible ripple shape) and `firmness` (how loose vs.
    packed the sand is) are independent knobs. Neither one is "how many
    centimeters it will sink" -- there's no such parameter here on purpose.
    How far a robot actually settles in is left for mujoco's contact physics
    to work out at simulation time, from firmness vs. that robot's actual
    weight on its wheels, exactly like real sand: a heavier robot or looser
    sand sinks deeper, a lighter robot or more packed sand sinks less.
    """

    field_size: tuple[float, float] = (4.0, 4.0)
    resolution: float = 0.05  # heightfield grid spacing in meters -- mesh detail only, doesn't affect how "sandy" it feels
    dune_height: float = 0.04  # visible ripple height (m); see class docstring
    num_waves: int = 10  # many small overlapping waves read as fine ripples, not a few big dunes
    # 0 = loose/dry sand (a dropped robot sinks in a lot), 1 = firm/packed
    # sand (barely any give). A material property, not a sink-depth preset.
    firmness: float = 0.4
    # radius around the origin smoothly flattened to ~0, so the robot never
    # spawns on a steep dune slope regardless of seed/field_size.
    spawn_clear_radius: float = 0.6
    seed: int = 0
    color: tuple[float, float, float, float] = (0.87, 0.72, 0.45, 1.0)
    # grainy speckle overlaid on `color` (via a random-mark texture) so the
    # surface reads as sand rather than a flat-colored hill.
    speckle_color: tuple[float, float, float] = (0.65, 0.52, 0.32)
    speckle_density: float = 0.4
    # sliding, torsional, rolling -- torsional is low (grains let a wheel
    # pivot easily) and rolling is raised (wheels plow through the sand).
    friction: tuple[float, float, float] = (0.7, 0.005, 0.3)
    # derived from firmness in __post_init__, not set directly
    solref: tuple[float, float] = field(init=False, repr=False)
    solimp: tuple[float, float, float, float, float] = field(init=False, repr=False)
    # kept in sync with dune_height so callers can uniformly do
    # `spawn_z = config.max_height + margin` across every terrain config type.
    max_height: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        firmness = min(max(self.firmness, 0.0), 1.0)
        # the whole hfield is raised by _SAND_HEADROOM (see add_sand_terrain),
        # so spawn clearance needs to account for both.
        self.max_height = self.dune_height + _SAND_HEADROOM
        # solref timeconst: kept in a narrow, empirically-checked range.
        # Above ~0.1 the contact spring becomes too weak to ever fully
        # arrest a falling robot within _SAND_HEADROOM's travel, so instead
        # of graduated sinking you get a cliff-edge fall straight through to
        # bedrock for almost any firmness. Small variation here still makes
        # packed sand feel a bit snappier than loose sand.
        timeconst = 0.03 + (1.0 - firmness) * 0.05
        self.solref = (timeconst, 1.0)
        # solimp dmin: minimum resistance right at first contact -- this is
        # the knob that actually does the graduated-sinking work here. Low
        # for loose sand (weight pushes in a lot before resistance builds
        # up -- at the extreme, firmness=0 is a quicksand-like full sink),
        # near mujoco's rigid default for packed sand (a few cm of settle,
        # like a real footstep in firm sand). width/dmax are fixed and just
        # shape how gradually resistance ramps in over the first ~10cm.
        dmin = 0.02 + firmness * 0.9
        self.solimp = (dmin, 0.95, 0.1, 0.5, 2.0)


@dataclass
class SlopeTerrainConfig:
    """A flat approach followed by one uniformly inclined ramp, for testing
    climbing/descending a constant grade rather than discrete bumps. The
    robot spawns on the flat part so it starts on level ground before
    reaching the incline.
    """

    field_size: tuple[float, float] = (6.0, 6.0)
    incline: float = 0.2  # radians, ramp tilts uphill along +y
    flat_length: float = 1.0  # distance from the origin (spawn point) to where the ramp starts, along +y
    # distinct colors so the flat-to-ramp transition is visible in the viewer
    flat_color: tuple[float, float, float, float] = (0.25, 0.3, 0.35, 1.0)
    ramp_color: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 1.0)
    friction: tuple[float, float, float] = (1.0, 0.1, 0.1)
    # kept at 0 so callers can uniformly do `spawn_z = config.max_height + margin`
    # across every terrain config type; a slope has no bump height of its own.
    max_height: float = field(default=0.0, init=False, repr=False)


TerrainConfig = VoxelTerrainConfig | SandTerrainConfig | SlopeTerrainConfig


def _add_box_grid_terrain(
    spec: mujoco.MjSpec,
    name: str,
    field_size: tuple[float, float],
    voxel_size: float,
    max_height: float,
    friction: tuple[float, float, float],
    color: tuple[float, float, float, float],
    origin: tuple[float, float],
    seed: int,
    solref: tuple[float, float] | None = None,
    solimp: tuple[float, float, float] | None = None,
) -> mujoco.MjsBody:
    """Add a grid of randomly-heighted box columns to `spec.worldbody`.

    field_size must be an integer multiple of voxel_size along each axis.
    Returns the terrain body that was added.
    """
    field_x, field_y = field_size

    n_x = round(field_x / voxel_size)
    n_y = round(field_y / voxel_size)
    if abs(n_x * voxel_size - field_x) > 1e-9 or abs(n_y * voxel_size - field_y) > 1e-9:
        raise ValueError(
            f"field_size {field_size} must be an integer multiple of voxel_size {voxel_size}"
        )

    rng = random.Random(seed)
    half_voxel = voxel_size / 2
    x_start = -field_x / 2 + half_voxel
    y_start = -field_y / 2 + half_voxel
    min_half_height = 1e-4  # mujoco boxes cannot have zero size

    extra_kwargs = {}
    if solref is not None:
        extra_kwargs["solref"] = list(solref)
    if solimp is not None:
        extra_kwargs["solimp"] = list(solimp)

    body = spec.worldbody.add_body(name=name, pos=[origin[0], origin[1], 0.0])
    for i in range(n_x):
        for j in range(n_y):
            half_height = max(rng.uniform(0.0, max_height) / 2, min_half_height)
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_voxel, half_voxel, half_height],
                pos=[x_start + i * voxel_size, y_start + j * voxel_size, half_height],
                group=1,
                friction=list(friction),
                rgba=list(color),
                **extra_kwargs,
            )
    return body


def add_voxel_terrain(
    spec: mujoco.MjSpec,
    config: VoxelTerrainConfig,
    origin: tuple[float, float] = (0.0, 0.0),
) -> mujoco.MjsBody:
    return _add_box_grid_terrain(
        spec,
        name="voxel_terrain",
        field_size=config.field_size,
        voxel_size=config.voxel_size,
        max_height=config.max_height,
        friction=config.friction,
        color=config.color,
        origin=origin,
        seed=config.seed,
    )


def _generate_dune_heightfield(
    nrow: int,
    ncol: int,
    field_size: tuple[float, float],
    seed: int,
    num_waves: int,
    spawn_clear_radius: float,
) -> np.ndarray:
    """Smooth, natural-looking dunes: a sum of a few sine waves at random
    frequency/phase/amplitude, normalized to [0, 1]. This is deliberately
    continuous (unlike the per-cell random heights in _add_box_grid_terrain)
    so it renders and behaves like rolling sand rather than discrete bumps.

    The dunes are smoothly flattened to ~0 within `spawn_clear_radius` of the
    origin so the robot never spawns on a steep dune slope by bad luck of
    the random seed.
    """
    field_x, field_y = field_size
    rng = np.random.default_rng(seed)
    row, col = np.mgrid[0:nrow, 0:ncol].astype(float)
    x = col / max(ncol - 1, 1)
    y = row / max(nrow - 1, 1)

    # Fractal-ish sum: each successive wave roughly doubles in frequency and
    # halves in amplitude, layering a gentle base undulation with
    # progressively finer ripples on top -- reads as wind-blown sand rather
    # than a few big smooth hills.
    height = np.zeros((nrow, ncol))
    freq = 1.5
    amplitude = 1.0
    for _ in range(num_waves):
        freq_x = freq * rng.uniform(0.8, 1.2)
        freq_y = freq * rng.uniform(0.8, 1.2)
        phase = rng.uniform(0.0, 2 * np.pi)
        height += amplitude * np.sin(2 * np.pi * (freq_x * x + freq_y * y) + phase)
        freq *= 1.8
        amplitude *= 0.55

    height -= height.min()
    peak = height.max()
    if peak > 1e-9:
        height /= peak

    if spawn_clear_radius > 0:
        x_m = (x - 0.5) * field_x
        y_m = (y - 0.5) * field_y
        dist = np.sqrt(x_m**2 + y_m**2)
        t = np.clip(dist / spawn_clear_radius, 0.0, 1.0)
        height *= t * t * (3.0 - 2.0 * t)  # smoothstep: 0 at the origin, 1 past the radius

    return height


def add_sand_terrain(
    spec: mujoco.MjSpec,
    config: SandTerrainConfig,
    origin: tuple[float, float] = (0.0, 0.0),
) -> mujoco.MjsBody:
    field_x, field_y = config.field_size
    ncol = max(round(field_x / config.resolution) + 1, 2)
    nrow = max(round(field_y / config.resolution) + 1, 2)
    heightfield = _generate_dune_heightfield(
        nrow, ncol, config.field_size, config.seed, config.num_waves, config.spawn_clear_radius
    )

    # sim/xml/world.xml has a shared invisible rigid floor at z=0 (a safety
    # net so nothing falls through gaps in other terrain types). If the sand
    # surface's own baseline also sat at z=0, wheels would hit that rigid
    # floor at the same moment they touch the sand, capping sinking to zero
    # regardless of how soft the contact is. Raising the whole hfield by
    # _SAND_HEADROOM opens up real room underneath the (unsunk) surface for
    # wheels to physically penetrate into before they hit bedrock -- how
    # much of that room actually gets used is up to the physics, not this.
    base_thickness = _SAND_HEADROOM + 0.2
    hfield_name = "sand_hfield"
    spec.add_hfield(
        name=hfield_name,
        nrow=nrow,
        ncol=ncol,
        size=[field_x / 2, field_y / 2, config.dune_height, base_thickness],
        userdata=heightfield.flatten().tolist(),
    )

    # Grainy speckle texture: a flat base color with randomly-placed marks,
    # tiled finely across the field. Without this the ground is just a
    # smoothly-colored bump -- the speckle is what actually reads as sand.
    tex_name = "sand_tex"
    mat_name = "sand_material"
    spec.add_texture(
        name=tex_name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mujoco.mjtMark.mjMARK_RANDOM,
        random=config.speckle_density,
        rgb1=list(config.color[:3]),
        markrgb=list(config.speckle_color),
        width=300,
        height=300,
    )
    material = spec.add_material(name=mat_name)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex_name
    material.texrepeat = [field_x * 4, field_y * 4]
    material.texuniform = True

    body = spec.worldbody.add_body(
        name="sand_terrain", pos=[origin[0], origin[1], _SAND_HEADROOM]
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield_name,
        group=1,
        friction=list(config.friction),
        material=mat_name,
        rgba=[1.0, 1.0, 1.0, config.color[3]],
        solref=list(config.solref),
        solimp=list(config.solimp),
        # Without this, mujoco averages solref/solimp with whatever the
        # other geom in each contact uses (the robot's wheels have no
        # custom values, i.e. mujoco's rigid defaults), which drowns out
        # the soft/sinking effect. A higher priority makes this geom's
        # contact params win outright instead of being averaged away.
        priority=1,
    )
    return body


def add_slope_terrain(
    spec: mujoco.MjSpec,
    config: SlopeTerrainConfig,
    origin: tuple[float, float] = (0.0, 0.0),
) -> mujoco.MjsBody:
    """Add a flat box from the -y field edge to `config.flat_length`, then a
    ramp box tilted by `config.incline` radians about the x-axis for the
    rest of the field. Both are finite boxes (not `type="plane"`, whose
    collision is an infinite half-space regardless of `size`), and the ramp
    is positioned so its near edge's top surface meets the flat box's top
    surface (z=0) exactly at y=flat_length -- a robot spawned at the origin
    starts on level ground before reaching the incline.
    """
    field_x, field_y = config.field_size
    flat_length = config.flat_length
    if not -field_y / 2 < flat_length < field_y / 2:
        raise ValueError(
            f"flat_length {flat_length} must leave room for both the flat part and the "
            f"ramp within field_size {config.field_size}"
        )

    half_thickness = 0.05
    incline = config.incline
    hx = field_x / 2

    body = spec.worldbody.add_body(name="slope_terrain", pos=[origin[0], origin[1], 0.0])

    flat_hy = (flat_length + field_y / 2) / 2
    flat_center_y = (flat_length - field_y / 2) / 2
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, flat_hy, half_thickness],
        pos=[0.0, flat_center_y, -half_thickness],
        group=1,
        friction=list(config.friction),
        rgba=list(config.flat_color),
    )

    ramp_hy = (field_y / 2 - flat_length) / 2
    # Align the ramp box's near-edge top surface (local point (0, -ramp_hy,
    # +half_thickness), rotated by `incline` about x) with the flat box's
    # top surface at (0, flat_length, 0).
    ramp_pos_y = flat_length + ramp_hy * math.cos(incline) + half_thickness * math.sin(incline)
    ramp_pos_z = ramp_hy * math.sin(incline) - half_thickness * math.cos(incline)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, ramp_hy, half_thickness],
        pos=[0.0, ramp_pos_y, ramp_pos_z],
        euler=[incline, 0, 0],
        group=1,
        friction=list(config.friction),
        rgba=list(config.ramp_color),
    )
    return body


def add_terrain(
    spec: mujoco.MjSpec,
    config: TerrainConfig,
    origin: tuple[float, float] = (0.0, 0.0),
) -> mujoco.MjsBody:
    """Dispatch to the right add_*_terrain() based on the config's type."""
    if isinstance(config, VoxelTerrainConfig):
        return add_voxel_terrain(spec, config, origin)
    if isinstance(config, SandTerrainConfig):
        return add_sand_terrain(spec, config, origin)
    if isinstance(config, SlopeTerrainConfig):
        return add_slope_terrain(spec, config, origin)
    raise TypeError(f"unsupported terrain config type: {type(config)!r}")
