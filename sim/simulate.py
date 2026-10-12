"""Passive-viewer simulation loop for a MuJoCo model."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer


def run(model: mujoco.MjModel, forward_speed: float = 0.0) -> None:
    """Step `model` in a viewer, driving both wheels forward at a constant
    angular velocity as a minimal smoke-test of the robot on the terrain.
    Robots without r_motor/l_motor (legged ones) hold their keyframe pose.
    """
    data = mujoco.MjData(model)

    names = {model.actuator(i).name for i in range(model.nu)}
    if "r_motor" in names and "l_motor" in names:
        data.ctrl[model.actuator("r_motor").id] = forward_speed
        data.ctrl[model.actuator("l_motor").id] = forward_speed
    elif model.nkey > 0:
        # legged robots (menagerie/*): start in the keyframe's standing joint
        # angles and hold them (qpos[:7] is the free base, kept at the spawn pose)
        data.qpos[7:] = model.key_qpos[0][7:]
        data.ctrl[:] = model.key_ctrl[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
