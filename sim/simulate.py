"""Passive-viewer simulation loop for a MuJoCo model."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer


def run(model: mujoco.MjModel, forward_speed: float = 0.0) -> None:
    """Step `model` in a viewer, driving both wheels forward at a constant
    angular velocity as a minimal smoke-test of the robot on the terrain.
    """
    data = mujoco.MjData(model)

    r_motor = model.actuator("r_motor").id
    l_motor = model.actuator("l_motor").id
    data.ctrl[r_motor] = forward_speed
    data.ctrl[l_motor] = forward_speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
