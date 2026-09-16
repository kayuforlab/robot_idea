# #地形の最大高さとその平均段差を求める．
# from make_xml import calc_average_step_height

# height_analysis = calc_average_step_height(0.01)

# print(height_analysis)

import json
import matplotlib.pyplot as plt
from make_xml import calc_average_step_height

# 入力値と average を保存するリスト
results = []

# 0.01 ～ 0.4 を 0.001 刻みで走査
start = 0.01
end = 0.4
step = 0.01

x_values = []
y_values = []

num_steps = int((end - start) / step) + 1

for i in range(num_steps):
    value = round(start + i * step, 3)

    # 関数実行
    height_analysis = calc_average_step_height(value)

    # average を取得
    average = height_analysis["average"]

    # 保存
    results.append({
        "max_height": value,
        "step_height_average": average
    })

    # グラフ用
    x_values.append(value)
    y_values.append(average)

    print(f"input={value:.3f}, average={average}")

# jsonl 保存
with open("average_results.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("jsonl saved: average_results.jsonl")

# グラフ作成
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values)

plt.xlabel("Max height [m]")
plt.ylabel("Step height average [m]")
plt.title("Input vs Average Step Height")
plt.grid(True)

# 保存
plt.savefig("average_graph.png", dpi=300)

# 表示
plt.show()

print("graph saved: average_graph.png")