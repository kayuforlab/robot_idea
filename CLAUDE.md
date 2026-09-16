# robot_idea

不整地を走破するロボットの設計案をMuJoCoでシミュレーションして検証するプロジェクト。

## 実行方法

```
uv run main.py
```

(または `.venv/bin/python main.py`)。`sim/xml/world.xml` (地面・空・カメラ) の上にランダムな凹凸地形を生成し、`ROBOT_NAME` で指定したロボットをスポーンさせ、MuJoCoのpassive viewerで動かす。`main.py` の `ROBOT_NAME` を変えることでロボットを切り替える。

### 既知の環境問題 (WSLg)

viewerプロセスは正常に起動しCPUも使っているのに、ウィンドウが画面に見えないことがある（タスクバーには出る）。WSLgがウィンドウを画面外の座標に生成するバグ。対処: タスクバーアイコンをクリックしてフォーカス→ `Win+↑` で最大化すると画面内に戻る。直らない場合は Windows側で `wsl --shutdown` してWSLを再起動。

## ディレクトリ構成

- `main.py` — エントリポイント。`ROBOT_NAME` を書き換えてシミュレートするロボットを選ぶ。
- `sim/` — ロボットに依存しない共通のシミュレーション基盤。
  - `sim/xml/world.xml` — ベースシーン（空・ライト・カメラ・衝突用の見えない地面）。ロボットも地形もここには含まれない。
  - `sim/terrain.py` — ランダムなボックスの列で不整地を生成する（`add_voxel_terrain`）。
  - `sim/world.py` — world.xml + 生成した地形 + ロボットのxmlを`MjSpec.attach`で合成して`MjModel`を作る。
  - `sim/simulate.py` — passive viewerでモデルを動かすループ（`r_motor`/`l_motor`に一定速度を入力するだけの簡易スモークテスト）。
- `<ROBOT_NAME>/xml/robot.xml` — 各ロボットの本体定義（**プロジェクトの規約**）。新しいロボットを追加するときはこの場所に置く。ロボット自身のメッシュ等のアセットも同じロボットのディレクトリ内に置き、他ロボットや`./xml`に依存させないこと（`sim/world.py`が`MjSpec.from_file`で単独ロードして`attach`するため、self-containedである必要がある）。
- `xml/` — **レガシーな試作物件**（Windows環境で書かれた旧プロトタイプ一式: MjSpec化される前の手書きXML、Inventorメッシュ、段差踏破の実験結果など）。**将来的に削除される予定**。現行の`sim/`パイプラインはここに依存しない設計にしてある（`sim/terrain.py`のdocstring参照）。新しいロボットを作るときは参考にするのは良いが、ファイルを直接参照・import してはいけない。必要なアセット（メッシュ等）は該当ロボットのディレクトリにコピーすること。

## ロボット規約

`sim/world.py`の`robot_xml_path()`が`./<ROBOT_NAME>/xml/robot.xml`を探す。このXMLは単独で`mujoco.MjSpec.from_file()`によりロード可能な完全なMuJoCoモデルである必要がある（`<worldbody>`直下にロボット本体、`<actuator>`に`r_motor`/`l_motor`という名前のアクチュエータ、`<asset>`に必要なメッシュなど）。ロボット本体のpos/ジョイントは原点基準で書く（world側でspawn位置にオフセットするフレームにattachされるため）。

`sim/simulate.py`の`run()`は`r_motor`・`l_motor`にちょうど一定速度を入れるだけなので、ロボットのXML側でこの2つの名前のアクチュエータを必ず定義すること。

## 現状のロボット: 2wheel_rover

左右2輪差動駆動、`./xml/base/*.xml`内の`2wheel_rover`ボディを元に作成。本体・ジョイント・アクチュエータ（`intvelocity`、kp=1）は共通で、ホイールのグロウザー（滑り止めラグ）形状だけが違う3種類を、それぞれ別のロボットフォルダとして分けてある（`xml/calc_max_height.py`が元々これらの形状ごとに踏破可能な最大段差を比較していたのに対応）:

- `2wheel_rover/normal/xml/robot.xml` — ラグなしの素のホイール（`mesh/wheel.stl`）
- `2wheel_rover/fan/xml/robot.xml` — 扇形ラグ（`mesh/fan.stl`）
- `2wheel_rover/one_way_grouser/xml/robot.xml` — 非対称の一方向ラグ（`mesh/one_way_grouser.stl`）

各フォルダは`./xml`から必要なメッシュをコピー済みで自己完結している。`main.py`の`ROBOT_NAME`をこれらのパス（例: `"2wheel_rover/one_way_grouser"`）に切り替えて使う。2輪しかないため前後方向に転倒しないよう、後方に引きずり式のスキッド（`stabilizer`ジオム）を共通で付けている。

`xml/calc_max_height.py` / `xml/max_height_step/` は、旧環境でこのホイール形状違いごとに踏破可能な最大段差を計測した実験の名残。同種の実験を今の`sim/`パイプラインでやる場合は、ここを参考にしつつ`sim/terrain.py`の`VoxelTerrainConfig`で段差を調整する。
