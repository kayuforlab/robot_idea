import mujoco
import mujoco.viewer
import platform
import time
import matplotlib.pyplot as plt
from collections import deque

# --- モデルロード ---
# os_name = platform.system()
# if os_name == "Windows":
#     model_path = r"C:\Users\moonshot\Desktop\Lab\research\RL\env\xml\sim_model.xml"
# elif os_name == "Linux":
#     model_path = r"C:\Users\moonshot\Desktop\Lab\research\RL\make_xml\sim_model.xml"

model_path = r"./env/xml/sim_model.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# --- ビューア初期化 ---
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

# --- ジョイント・アクチュエータID ---
r_motor_id = model.actuator("r_motor").id
l_motor_id = model.actuator("l_motor").id
r_wheel_id = model.joint("r_wheel_joint").id
l_wheel_id = model.joint("l_wheel_joint").id

# --- グラフ初期化 ---
plt.ion()  # インタラクティブモードON
fig, ax = plt.subplots()
WINDOW = 200  # 表示する最新の200ステップ
r_wheel_history = deque([0]*WINDOW, maxlen=WINDOW)

line, = ax.plot(range(WINDOW), r_wheel_history)
ax.set_ylim(-4, 4)  # 角速度の範囲に応じて調整
ax.set_xlabel("ステップ")
ax.set_ylabel("r_wheel角速度 [rad/s]")

# --- シミュレーションループ ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # --- 角速度取得 ---
        r_wheel_v = data.qvel[r_wheel_id]

        # --- データ更新 ---
        r_wheel_history.append(r_wheel_v)
        line.set_ydata(r_wheel_history)

        # --- グラフ更新 ---
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.00001)

        # --- MuJoCoステップ ---
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
