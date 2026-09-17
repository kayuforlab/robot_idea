# robot_idea

不整地を走破するロボットの設計案を考え、MuJoCoでシミュレーションして検証するプロジェクトである。
実行方法・ディレクトリ構成などは [CLAUDE.md](CLAUDE.md) を参照。

## ホイール形状（2wheel_rover）

共通シャーシに対しラグ形状だけが違う3種（タイヤ外径・幅は共通、`ROBOT_NAME`で切替）。

| | normal | fan | one_way_grouser |
|---|---|---|---|
| ラグ | なし（滑らかな円筒） | 扇形×10枚、ハブ〜外周まで連続 | 直進リブ×10本、外周のみ突出 |
| 断面対称性 | 対称 | 対称 | 非対称（片側急勾配のラチェット状） |
| 特徴 | 転がり抵抗最小。ベースライン | 砂・礫を掻き込むパドル的グリップ。硬地面では抵抗増 | 想定回転方向の段差食い込みに強い。逆回転・平滑地では弱い |

- [2wheel_rover/normal](2wheel_rover/normal/xml/robot.xml)
- [2wheel_rover/fan](2wheel_rover/fan/xml/robot.xml)
- [2wheel_rover/one_way_grouser](2wheel_rover/one_way_grouser/xml/robot.xml)
