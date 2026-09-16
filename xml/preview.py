import mujoco
import mujoco.viewer
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os


model_path = r"./env/xml/sim_model.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)



scene_option=mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False




def get_terrain_height_map(model, data, robot_pos, out_path, grid_size=0.5, resolution=120, make_graph=False):
    """
    世界座標系でロボット周囲の地形をスキャンし、グレースケール画像として保存する。
    基準高さは世界座標 Z=0 とし、取得した全領域を画像に反映する。
    """
    # --- 1. 地形データの取得 ---
    lin_x = np.linspace(robot_pos[0] - grid_size/2, robot_pos[0] + grid_size/2, resolution)
    lin_y = np.linspace(robot_pos[1] - grid_size/2, robot_pos[1] + grid_size/2, resolution)
    
    x_grid, y_grid = np.meshgrid(lin_x, lin_y, indexing='ij')
    relative_height_map = np.zeros((resolution, resolution))
    height_map = np.zeros((resolution, resolution))
    
    # レイの発射開始高さ（ロボットより高く、地形を確実にカバーする位置）
    ray_start_z = 1 
    ray_direction = np.array([0, 0, -1], dtype=np.float64)
    geom_id_arr = np.zeros(1, dtype=np.int32)#衝突したgeomのid格納用
    #note 左からgeom number 0,1,2,.... group1は地形,group2はローバ
    predict_geom = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8) 
    for i in range(resolution):
        for j in range(resolution):
            p_start = np.array([x_grid[i, j], y_grid[i, j], ray_start_z], dtype=np.float64)
            dist = mujoco.mj_ray(
                model, 
                data, 
                p_start, #rayのstart地点
                ray_direction, 
                predict_geom, #検知対象グループ
                1, #静止しているものを対象とする
                -1, #除外するbody ID
                geom_id_arr)
            if dist >= 0:
                _height = ray_start_z - dist
                height_map[i,j] = _height
                relative_height_map[i, j] = _height - robot_pos[2]#ロボットからの相対的な高さにする．
            else:
                height_map[i, j] = 0.0 
                relative_height_map[i, j] = 0.0

    ########################################
    # グラフ作成
    ########################################
    if make_graph:
        #height mapの色分けgrid
        fig_save = plt.figure(figsize=(6, 6))
        ax_save = fig_save.add_subplot(111)
        half_size = grid_size / 2.0
        ext = [-half_size, half_size, -half_size, half_size]
        limit = 0.5  # 表示したい高さの範囲
        
        img = ax_save.imshow(
            relative_height_map.T, 
            extent=ext, 
            origin='lower', 
            cmap='seismic', 
            vmin=-limit,
            vmax=limit
        )
        fig_save.colorbar(img, ax=ax_save, label='Height relative to rover body [m]')
        ax_save.set_xlabel("Relative X (m)")
        ax_save.set_ylabel("Relative Y (m)")
        _path = out_path + r"relative_height_map.png"
        fig_save.savefig(_path)
        plt.close(fig_save)

        #####立体化######
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        x_range = np.linspace(-half_size, half_size, resolution)
        y_range = np.linspace(-half_size, half_size, resolution)
        X, Y = np.meshgrid(x_range, y_range)
        surf = ax_3d.plot_surface(
            X, Y, relative_height_map.T, 
            cmap='seismic', 
            vmin=-limit, 
            vmax=limit,
            linewidth=0, 
        )
        ax_3d.set_zlim(-limit, limit)
        ax_3d.set_xlabel('Relative X (m)')
        ax_3d.set_ylabel('Relative Y (m)')
        ax_3d.set_zlabel('Relative height (m)')
        ax_3d.set_title('3D Terrain Surface')
        ax_3d.view_init(elev=50, azim=-100) #方向
        _path = out_path + r"relative_height_map_3d.png"
        fig_3d.savefig(_path)
        plt.close(fig_3d)

        ####ただのheight map####
        half_size = grid_size / 2.0
        ext = [robot_pos[0] - half_size, robot_pos[0] + half_size, 
            robot_pos[1] - half_size, robot_pos[1] + half_size]

        # 表示する高さの絶対範囲
        z_min = 0.0
        z_max = 1.0 
        cmap_abs = 'cividis' 
        ##### 1. 2D Height Map (絶対高さ) #####
        fig_save = plt.figure(figsize=(6, 6))
        ax_save = fig_save.add_subplot(111)
        img = ax_save.imshow(
            height_map.T, 
            extent=ext, 
            origin='lower', 
            cmap=cmap_abs, 
            vmin=z_min, 
            vmax=z_max
        )
        fig_save.colorbar(img, ax=ax_save, label='Absolute Height Z (m)')
        ax_save.set_xlabel("World X (m)")
        ax_save.set_ylabel("World Y (m)")
        ax_save.set_title("Absolute Terrain Height Map")
        _path_2d = out_path + r"absolute_height_map.png"
        fig_save.savefig(_path_2d)
        plt.close(fig_save)


        ##### 2. 3D 立体化 (絶対高さ) #####
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        x_range = np.linspace(ext[0], ext[1], resolution)
        y_range = np.linspace(ext[2], ext[3], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        surf = ax_3d.plot_surface(
            X, Y, height_map.T, 
            cmap=cmap_abs, 
            vmin=z_min, 
            vmax=z_max,
            linewidth=0, 
            antialiased=True # フラグを修正
        )
        ax_3d.set_zlim(z_min, z_max)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Absolute Height Z (m)')
        ax_3d.view_init(elev=50, azim=-100) 
        _path_3d = out_path + r"absolute_height_map_3d.png"
        fig_3d.savefig(_path_3d)
        plt.close(fig_3d)

        #dataの保存
        _csv_path = out_path +r"csv/"
        os.makedirs(_csv_path, exist_ok=True)
        _path = _csv_path + r"relative_height_map.csv"
        np.savetxt(_path.replace(".png", ".csv"), relative_height_map, delimiter=",", fmt="%.6f")
        _path = _csv_path + r"height_map.csv"
        np.savetxt(_path.replace(".png", ".csv"), relative_height_map, delimiter=",", fmt="%.6f")
    
    return height_map




with mujoco.viewer.launch_passive(model, data) as viewer:

    ########## main cam
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
    if cam_id == -1:
        raise ValueError("XML に camera name='main_cam' が存在しません")
    # viewer のカメラを切り替え
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = cam_id
    ####################


    # viewer に設定を反映，主にジョイントを表示するため

    #Global座標系表示
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

    #メッシュを凸包化する
    vopt = mujoco.MjvOption()
    # vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1

    imu_acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "IMU")
    imu_acc_adr = model.sensor_adr[imu_acc_id]
    imu_acc_dim = model.sensor_dim[imu_acc_id]   

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "GYRO")
    gyro_adr = model.sensor_adr[gyro_id]
    gyro_dim = model.sensor_dim[gyro_id]

    vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "VELOCITY")
    vel_adr = model.sensor_adr[vel_id]
    vel_dim = model.sensor_dim[vel_id]        

    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_geom")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_geom")

    r_motor_id = model.actuator("r_motor").id
    l_motor_id = model.actuator("l_motor").id

    r_wheel_id = model.joint("r_wheel_joint").id
    l_wheel_id = model.joint("l_wheel_joint").id

    r_qposadr = model.jnt_qposadr[r_wheel_id]   # qpos の開始インデックス
    r_qveladr = model.jnt_dofadr[r_wheel_id]    # qvel の開始インデックス

    l_qposadr = model.jnt_qposadr[l_wheel_id]   # qpos の開始インデックス
    l_qveladr = model.jnt_dofadr[l_wheel_id]    # qvel の開始インデックス

    # print("R,L motor ID : ",r_motor_id,l_motor_id)
    # print("R,L wheel ID : ",r_wheel_id,l_wheel_id)
    # print("")

    #可視化
    # ---------- Matplotlib 設定 ----------
    # plt.ion()
    # fig, ax = plt.subplots()
    # ax.set_title("IMU Angular Velocity (GYRO)")
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Angular velocity [rad/s]")

    # # 表示範囲3s
    # window = 3.0
    # timestep = model.opt.timestep
    # max_len = int(window / timestep)

    # time_buf = deque(maxlen=max_len)
    # wx_buf = deque(maxlen=max_len)
    # wy_buf = deque(maxlen=max_len)
    # wz_buf = deque(maxlen=max_len)

    # (line_wx,) = ax.plot([], [], label="wx")
    # (line_wy,) = ax.plot([], [], label="wy")
    # (line_wz,) = ax.plot([], [], label="wz")
    # ax.legend()
    # plt.tight_layout()
    _path = r"./env/out/"
    os.makedirs(_path, exist_ok=True)
    get_terrain_height_map(model,data,[2,2,0.3],out_path=_path,make_graph=False)    
    max_r, max_l = 0, 0
    min_r, min_l = 0, 0
    steps = 0
    ROUND_NUM = 4
    #慣性系を表示
    # viewer.opt.flags[:] = scene_option.flags[:]
    start_time = time.time()
    while viewer.is_running():
        cam = viewer.cam
        mujoco.mj_step(model, data)
        viewer.sync()
        # print("Camera位置:", cam.lookat,"水平角度:", cam.azimuth,"垂直角度:", cam.elevation)

        r_wheel_v = round(data.qvel[r_qveladr],ROUND_NUM)#角速度取得[rad/s]
        l_wheel_v = round(data.qvel[l_qveladr],ROUND_NUM)
        r_motor_torque = round(data.actuator_force[r_motor_id],ROUND_NUM)
        l_motor_torque = round(data.actuator_force[l_motor_id],ROUND_NUM)
        r_wheel_angle = round(data.qpos[r_qposadr], ROUND_NUM)
        l_wheel_angle = round(data.qpos[l_qposadr], ROUND_NUM)
        # print(f"角度 [rad]： R={r_wheel_angle} , L={l_wheel_angle}")
        # print("角速度：",round(r_wheel_v,ROUND_NUM)," , ",round(l_wheel_v,ROUND_NUM), "トルク：",round(r_motor_torque,ROUND_NUM)," , ",round(l_motor_torque,ROUND_NUM))
        # print("角速度：",r_wheel_v)

        imu_acc = data.sensordata[imu_acc_adr: imu_acc_adr + imu_acc_dim]
        gyro = data.sensordata[gyro_adr: gyro_adr + gyro_dim]
        vel = data.sensordata[vel_adr: vel_adr + vel_dim]

        # print(f"IMU 加速度: {imu_acc},角速度{gyro}")
        # print(gyro)
        # print(vel)

        #可視化
        # t = time.time() - start_time
        # time_buf.append(t)
        # wx_buf.append(gyro[0])
        # # wx_buf.append(r_motor_torque)
        # # wy_buf.append(r_wheel_v)
        # wy_buf.append(gyro[1])
        # wz_buf.append(gyro[2])
        # # wz_buf.append(0)
        # # # ---------- プロット更新 ----------
        # line_wx.set_data(time_buf, wx_buf)
        # line_wy.set_data(time_buf, wy_buf)
        # line_wz.set_data(time_buf, wz_buf)
        # # x軸を3秒幅でスライド表示
        # if len(time_buf) > 0:
        #     ax.set_xlim(max(0, time_buf[-1] - window), time_buf[-1])
        # ax.relim()
        # ax.autoscale_view(scalex=False, scaley=True)
        # plt.pause(0.001)

        
        if steps == 1000:
            max_r, max_l = 0, 0
            min_r, min_l = 1, 1

        r_pos = np.round(data.geom_xpos[r_wheel_geom_id].copy(), 3)
        l_pos = np.round(data.geom_xpos[l_wheel_geom_id].copy(), 3)
        if max_r<r_pos[2]:
            max_r = r_pos[2]
        if max_l<l_pos[2]:
            max_l = l_pos[2]
        if min_r>r_pos[2]:
            min_r = r_pos[2]
        if min_l>l_pos[2]:
            min_l = l_pos[2]
        print("b :", r_pos, l_pos, max_r, max_l, min_r, min_l)

    

        # viewer.sync()

        # save_terrain_height_grid_image(model,data,[2,2,0.3])    

        time.sleep(model.opt.timestep)  # 実時間に近づける
        steps += 1