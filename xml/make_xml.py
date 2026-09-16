import mujoco as mj
import random
import platform
os_name = platform.system()
import numpy as np
import matplotlib.pyplot as plt
import statistics
import json
import re

#通常
# par_path = "rl/jsons/env_parameters.jsonc"
#LSTM
par_path = "./sac/rl_t/jsons/env_parameters_t_.jsonc"


#お試しで地形作るとき
max_height_from_main = 0.001
# max_height_from_main = None

SEED = 3


def load_jsonc(path):
    """コメント付きJSONCファイルを読み込んでdictとして返す"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # コメント削除（//... と /* ... */ の両方）
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # クォートなしのキーを自動でダブルクォート化
    # 例:  field_seed: 1  →  "field_seed": 1
    text = re.sub(r'(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):', r'\1"\2"\3:', text)

    return json.loads(text)

# if os_name == "Windows":
#   with open(r"C:\Users\moonshot\Desktop\Lab\research\RL\env\env_parameters.jsonc", "r") as f:
#     par = json.load(f)
#   spec = mj.MjSpec.from_file(r"C:\Users\moonshot\Desktop\Lab\research\RL\env\xml\base_model.xml")
# elif os_name == "Linux":
#   with open("data.jsonc", "r") as f:
#     data = json.load(f)
#   spec = mj.MjSpec.from_file(r"C:\Users\moonshot\Desktop\Lab\research\RL\env\xml\base_model.xml")


def terrain_height_analysis(height_data):
  diff_up_down = abs(height_data[:-1, :] - height_data[1:, :])
  diff_left_right = abs(height_data[:, :-1] - height_data[:, 1:])
  data = np.concatenate([diff_up_down.ravel(), diff_left_right.ravel()])
  print(f"平均：{data.mean()}")
  print(f"標準偏差：{np.std(data)}")
  print(f"中央値：{statistics.median(data)}")

  result = {
     "average":data.mean(),
     "std":np.std(data),
     "center":statistics.median(data)
  }
  # print(data)


  # # フォント設定
  # plt.rcParams['font.family'] = 'Times New Roman'
  # plt.figure()
  # plt.hist(
  #     data,
  #     bins=100,
  #     color='steelblue',   # 好きな色に変更可
  #     edgecolor='black',
  #     label='Step distribution'
  # )
  # plt.xlabel("Step height [m]", fontsize=16)
  # plt.ylabel("Count [-]", fontsize=16)
  # plt.xlim(0, 0.1)
  # plt.ylim(bottom=0)
  # plt.xticks(fontsize=14)
  # plt.yticks(fontsize=14)
  # # 凡例
  # plt.legend(fontsize=14)
  # plt.show()
  

  return result






def create_boxy_terrain(spec, field_length, cube_length, max_height, seed=None):
  #seed値を設定
  if seed is None:
    print("seed値を設定してください")
    return
  else:
     random.seed(seed)
  # print("地形寸法[m]：",field_length)
  # print("ブロックの大きさ[m]：",cube_length)
  print(f"最大高さ[m]：{max_height}, box幅{cube_length}で地形生します. seed:{seed}")
  # print("seed値 : ", seed, " で地形生成します")

  #地形全体の半辺長
  FIELD_HALF_LENGTH_X = field_length[0]/2
  FIELD_HALF_LENGTH_Y = field_length[1]/2
  #1つの立方体の半辺長
  CUBE_HALF_LENGTH = cube_length/2
  #x,yに並べる個数
  GRID_SIZE_X = FIELD_HALF_LENGTH_X / CUBE_HALF_LENGTH
  GRID_SIZE_Y = FIELD_HALF_LENGTH_Y / CUBE_HALF_LENGTH
  if GRID_SIZE_X.is_integer() and GRID_SIZE_Y.is_integer():
     GRID_SIZE_X = int(GRID_SIZE_X)
     GRID_SIZE_Y = int(GRID_SIZE_Y)
  else:
     print("ブロックの個数は整数である必要があります")
     print("ブロックの個数 x,y ：", GRID_SIZE_X, GRID_SIZE_Y)
     return 
  
  #立方体一個の全長
  STEP = cube_length
  #色
  # BROWN = [0.460, 0.362, 0.216, 1.0]
  COLOR = [0.4, 0.3, 0.2, 1.0]

#   if spec == None:
#     spec=mj.MjSpec()

  # 生成するジオメトリのデフォルト形状を設定
  main = spec.default
  main.geom.type = mj.mjtGeom.mjGEOM_BOX

  #地形用のbodyを追加．位置は頂点がglobal原点に来るようにする
#   body = spec.worldbody.add_body(pos=[0,0,0], name=name)
  body = spec.worldbody.add_body(pos=[FIELD_HALF_LENGTH_X,FIELD_HALF_LENGTH_Y,0],name="boxy terrain")

  x_beginning = -FIELD_HALF_LENGTH_X + CUBE_HALF_LENGTH
  y_beginning = FIELD_HALF_LENGTH_Y - CUBE_HALF_LENGTH
  height_list = np.zeros([GRID_SIZE_X,GRID_SIZE_Y])
  for i in range(GRID_SIZE_X):
    for j in range(GRID_SIZE_Y):
      
      # z = np.random.randn()
      # # シグモイド変換（0,1）
      # x = 1 / (1 + np.exp(-z))
      # cube_half_height = x*max_height/2#正規分布

      cube_half_height = random.uniform(0,1)*max_height/2#mujocoでbox作るときの指定が長さの半分だから
      height_list[i,j] = cube_half_height * 2
      body.add_geom(
        size=[CUBE_HALF_LENGTH, CUBE_HALF_LENGTH,cube_half_height],
        pos=[x_beginning+i*STEP, y_beginning-j*STEP, cube_half_height],
        group=1,
        friction=[1, 0.1, 0.1],
        rgba=COLOR,
      )
    
  height_analysis = terrain_height_analysis(height_list)

  print("地形生成終了")
  return height_analysis



def update_terrain(terrain_seed,terrain_level,env_par,base_model_path=None):

  # par = load_jsonc("./env/env_parameters.jsonc")
  par = env_par
  ############# parameters #############
  """
  EXPERIMENTAL_FIELD_LENGTH：実験フィールド自体の大きさ[m]
  FIELD_LENGTH：ロボットとターゲットの存在するフィールドの大きさ[m]
  CUBE_LENGTH：ブロックの一辺の長さ[m]
  MAX_HEIGHT：ブロックの最大高さ[m]
  """
  EXPERIMENTAL_FIELD_LENGTH = par["experimental_field_length"]
  # FIELD_LENGTH = par["field_length"]
  CUBE_LENGTH = par["cube_length"]
  if terrain_level != None:
    MAX_HEIGHT = terrain_level
  else:
    MAX_HEIGHT = par["max_height"]


  #######################################

  if base_model_path==None:
    spec = mj.MjSpec.from_file(r"./env/xml/base_model.xml")
  elif base_model_path!=None:
    spec = mj.MjSpec.from_file(base_model_path)

  geoms = spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
  
  height_analysis = create_boxy_terrain(spec=spec, field_length=EXPERIMENTAL_FIELD_LENGTH, cube_length=CUBE_LENGTH, max_height=MAX_HEIGHT, seed=terrain_seed)
  model = spec.compile()
  output_xml = spec.to_xml()

  #出力したxmlを保存
  # if os_name == "Windows":
  #   with open(r"C:\Users\moonshot\Desktop\Lab\research\RL\env\xml\sim_model.xml",'w') as f:
  #       f.write(output_xml)
  # if os_name == "Linux":
  #   with open(r"C:\Users\moonshot\Desktop\Lab\research\RL\env\xml\sim_model.xml",'w') as f:
  #       f.write(output_xml)

  with open(r"./env/xml/sim_model.xml",'w') as f:
      f.write(output_xml)
  return 0



#平均最大高さを調べたいだけの関数．
def calc_average_step_height(max_height):
   
  par = load_jsonc(par_path)
  ############# parameters #############
  EXPERIMENTAL_FIELD_LENGTH = par["experimental_field_length"]
  # FIELD_LENGTH = par["field_length"]
  CUBE_LENGTH = par["cube_length"]
  if max_height == None:
    MAX_HEIGHT = par["max_height"]
  else:
    MAX_HEIGHT = max_height
  SEED = par["terrain_seed"]
  SEED = 0
  #######################################

  spec = mj.MjSpec.from_file(r"./env/xml/base_model.xml")
  geoms = spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)

  height_analysis = create_boxy_terrain(spec=spec, field_length=EXPERIMENTAL_FIELD_LENGTH, cube_length=CUBE_LENGTH, max_height=MAX_HEIGHT, seed=SEED)


  return height_analysis




def main(max_height):
  
  
  par = load_jsonc(par_path)
        
  ############# parameters #############
  """
  EXPERIMENTAL_FIELD_LENGTH：実験フィールド自体の大きさ[m]
  FIELD_LENGTH：ロボットとターゲットの存在するフィールドの大きさ[m]
  CUBE_LENGTH：ブロックの一辺の長さ[m]
  MAX_HEIGHT：ブロックの最大高さ[m]
  """
  EXPERIMENTAL_FIELD_LENGTH = par["experimental_field_length"]
  # FIELD_LENGTH = par["field_length"]
  CUBE_LENGTH = par["cube_length"]
  if max_height == None:
    MAX_HEIGHT = par["max_height"]
  else:
    MAX_HEIGHT = max_height
  # SEED = par["terrain_seed"]

  #######################################



  spec = mj.MjSpec.from_file(r"./env/xml/base_model.xml")
  geoms = spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
  # print(geoms)

  # import numpy as np
  # for h in np.arange(0, 0.5 + 1e-9, 0.05):
  #   print(h)
  #   make_terrain = create_boxy_terrain(spec=spec, field_length=EXPERIMENTAL_FIELD_LENGTH, cube_length=CUBE_LENGTH, max_height=h, seed=SEED)

  height_analysis = create_boxy_terrain(spec=spec, field_length=EXPERIMENTAL_FIELD_LENGTH, cube_length=CUBE_LENGTH, max_height=MAX_HEIGHT, seed=SEED)
  model = spec.compile()
  output_xml = spec.to_xml()

  #出力したxmlを保存
  # if os_name == "Windows":
  #   with open(r"C:\Users\moonshot\Desktop\Lab\research\RL\env\xml\sim_model.xml",'w') as f:
  #       f.write(output_xml)
  # if os_name == "Linux":
  #   with open(r"C:\Users\moonshot\Desktop\Lab\research\RL\env\xml\sim_model.xml",'w') as f:
  #       f.write(output_xml)

  with open(r"./env/xml/sim_model.xml",'w') as f:
      f.write(output_xml)
  return 0



if __name__ == "__main__":
    main(max_height_from_main)





