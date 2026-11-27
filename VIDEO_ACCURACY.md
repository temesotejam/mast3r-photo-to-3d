# 単眼動画から動く部品を復元する

同じ脚が見え隠れする動画では、脚がよく見えた時刻に写った模様を、後で脚が重なった時刻にも利用できます。このアプリは、最初のコマで囲んだ部品をそれぞれ剛体として扱い、観測できた面を部品座標系に蓄積します。

## 映像から向きを推定する手がかり

| 読んだ論文 | この復元との関係 |
|---|---|
| Wen ほか、BundleSDF（CVPR 2023） | 遮蔽後に再発見した物体を、保存しておいたキーフレームの特徴と照合します。原論文はRGB-Dカメラを使います。本実装はRGB動画の前後のコマから参照候補を取り、奥行きの初期値をMASt3Rで求めます。[本文](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf)・[補足 §1](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wen_BundleSDF_Neural_6-DoF_CVPR_2023_supplemental.pdf) |
| Engel ほか、Direct Sparse Odometry（IEEE TPAMI 2018） | 位置合わせに画素の明るさを直接使います。本実装では複数コマで確認できた平らな面の模様を候補姿勢で元映像に投影し、色のずれを減らします。計算から取り分けておいた別の画像領域でも照合します。[本文 §2](https://cvai.cit.tum.de/_media/spezial/bib/engel_et_al_pami2018.pdf) |
| Schönberger ほか、Pixelwise View Selection for Unstructured Multi-View Stereo（ECCV 2016） | よく似た画像でも、同じ方向から見ていると奥行きを決める視差が足りません。論文は画素の見た目、三角測量の角度、再投影の整合性を使って参照画像を選びます。[本文 §4](https://demuc.de/papers/schoenberger2016mvs.pdf) |
| Yang ほか、LASR（CVPR 2021） | 1本のRGB動画に対し、推定した立体から生成した輪郭・画像上の動き・画素値を元映像と照合して形状と関節運動を求めます。本アプリの輪郭評価は、推定した部品を元の時刻に投影して正解輪郭と比べるものです。[本文 §3](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_LASR_Learning_Articulated_Shape_Reconstruction_From_a_Monocular_Video_CVPR_2021_paper.pdf) |
| Seitz・Dyer、Photorealistic Scene Reconstruction by Voxel Coloring（IJCV 1999） | 同じ表面を見ている視点間で色が整合するかを調べます。本実装では一度復元した点を各時刻に投影し、見えるはずの複数のコマで色が食い違う点を取り除きます。手前に別の面がある場合や二つの部品のマスクが重なった場合、その画素での色判定は保留します。[本文 §3–4](https://www.cs.columbia.edu/~allen/PHOTOPAPERS/seitz.long.pdf) |
| Kopf ほか、Robust Consistent Video Depth Estimation（CVPR 2021） | 単眼モデルの奥行き推定を時刻間の幾何に合わせる考え方を参考にしました。本実装では、部品上の広い範囲で光学フローと特徴点照合が同じ姿勢を支持し、奥行きだけがずれた場合に部品ごとの倍率を求めます。[本文](https://robust-cvd.github.io/) |
| Karaev ほか、CoTracker3（ICCV 2025） | 公式のオフライン追跡器を実際に動かし、点の追跡結果を3D姿勢の計算に渡す試験も行いました。同じMASt3R奥行きによるローカル比較では、公開デモの奥の部品は5.49°から4.36°、異なる動きの奥の部品は19.07°から22.56°でした。公開版は上記の複数コマの色照合を使用しています。[論文・公式実装](https://github.com/facebookresearch/co-tracker) |

動画1本から動く立体を自由視点で見る研究としては、Wang ほかの **Shape of Motion**（ICCV 2025）が目的に近いものです。全編にわたって同じ3D点を持ち、点が共通の剛体運動にどの程度従うかを推定します。投影した色・奥行き・画面上の軌跡を入力と比較して動画ごとに最適化します。今の公開版は最初に指定した剛体部品ごとの点群を使っており、Shape of Motion本体の実行は今後の比較対象です。[本文 §3、§5](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.pdf)

## 正解が分かる動画での評価

- **回転誤差**は初期姿勢から各時刻までの3D回転について、正解と推定との間の角度を度数で表し、二乗平均平方根を求めます。姿勢を補間した時刻も集計し、観測・補間・追跡不可の件数を併記します。
- **輪郭のIoU**は、各部品を元のカメラに投影した点群と正解輪郭が重なる面積を、両者を合わせた面積で割った値です。**厚み／長さ**は点群を初期姿勢の既知の部品軸に移し、各方向の1〜99パーセンタイルの幅で比較します。
- **見えた表面の精度と再現率**には、入力のいずれかの時刻で実際に見えた正解の表面だけを使います。単眼の原点とスケールの違いを取り除くため、部品ごとに移動と一様倍率を合わせ、回転は固定します。部品長の2%以内に正解面がある復元点の割合を精度、逆に復元点がある正解点の割合を再現率とし、調和平均をFスコアとします。実寸や部品間の距離はこの指標の対象外です。[距離から精度・再現率を計算する手法の原典](https://www.tanksandtemples.org/tutorial/)

正解の姿勢、奥行き、部品ラベルは評価にだけ使います。GitHub Actionsの公開デモでは同じ動画・初期指定が以前に公開されている場合、前後の数値を同じ定義で比較できます。[デモを再生](https://temesotejam.github.io/mast3r-photo-to-3d/video.html)・[各コマの誤差と3D表面の評価](https://temesotejam.github.io/mast3r-photo-to-3d/video-example/quality.json)

## 公開デモでの結果

GitHub ActionsのCPUで公開デモの16コマを最初から復元し、前回公開した同じ動画の復元結果と比較しました。各コマの正解は復元後の採点だけに使用しています。[GitHub Actionsの実行記録](https://github.com/temesotejam/mast3r-photo-to-3d/actions/runs/46010815104)

| 部品 | 3D回転RMSE（旧 → 今回） | 輪郭IoU（旧 → 今回） | 見えた表面のFスコア（旧 → 今回） | 厚み／長さ（旧 → 今回、正解） |
|---|---:|---:|---:|---:|
| 奥の青緑 | 5.37° → 1.97° | 0.847 → 0.843 | 0.708 → 0.700 | 0.179 → 0.171（0.132） |
| 手前の橙 | 0.27° → 0.27° | 0.817 → 0.851 | 0.780 → 0.783 | 0.125 → 0.122（0.127） |

奥の部品は14コマで観測し、2コマを補間しました。約97%隠れた2.00秒のコマは、補間した向きの誤差が10.35°から1.74°になりました。手前の部品は16コマすべてで観測しました。

## 別の動きと撮影条件での結果

次の表は、公開デモとは形、表面の模様、カメラ条件、動きが異なる合成動画でのローカル比較です。同じ動画、先頭コマの枠、同じMASt3Rの推論結果を旧版と今回の版に与えました。14コマの動画は実装を確定するまで評価に使っていません。

| 動画・部品 | 3D回転RMSE（旧 → 今回） | 輪郭IoU（旧 → 今回） | 見えた表面のFスコア（旧 → 今回） | 今回の厚み／長さ（正解） |
|---|---:|---:|---:|---:|
| 動きを変えた12コマ・緑 | 19.07° → 2.42° | 0.630 → 0.603 | 0.516 → 0.537 | 0.359（0.152） |
| 同・ピンク | 2.18° → 2.18° | 0.784 → 0.801 | 0.561 → 0.563 | 0.199（0.184） |
| 未使用の14コマ・奥 | 66.09° → 66.09° | 0.573 → 0.643 | 0.175 → 0.186 | 1.071（0.131） |
| 同・手前 | 12.31° → 12.31° | 0.701 → 0.743 | 0.667 → 0.692 | 0.434（0.203） |

14コマの動画では奥の部品が少なくとも32%見えていて、全コマに姿勢が付きました。それでも向きと厚みには大きな誤りがあります。別途、生成時の水平画角を指定しても、今回の版の奥の回転誤差は51.03°残りました。

同じ14コマを再生成して評価できます。最初のコマの枠を指定して復元し、得られた点群と姿勢を最後に正解と照合します。

```bash
python scripts/make_video_holdout.py --output-dir /tmp/mast3r-holdout --reference-arrays /tmp/mast3r-holdout/reference.npz
python reconstruct_video.py --video /tmp/mast3r-holdout/video.mp4 --parts /tmp/mast3r-holdout/parts.json --camera fixed --output /tmp/mast3r-holdout/scene.json
python scripts/video_quality.py /tmp/mast3r-holdout/truth.json /tmp/mast3r-holdout/scene.json --reference /tmp/mast3r-holdout/reference.npz
```

## 実写の竜の模型での評価

[BOPのTUD-Lデータセット](https://bop.felk.cvut.cz/datasets/)から、人物が竜の模型を手で動かす実写シーケンス `test/000001` の、重複しない2区間を使いました。入力はRGB動画、最初の1コマの枠、撮影時の焦点距離から求めた水平画角63.71°です。深度センサー、モデル、2コマ目以降の正解姿勢・マスクは復元に渡していません。回転は初期姿勢からの相対回転をデータセットの注釈と照合しています。

| 元データのコマ番号 | 入力コマ数 | 旧版 | 今回 |
|---|---:|---|---|
| 0から88まで8コマおき | 12 | 初期コマに続く姿勢が得られず、出力なし | 12コマに姿勢が付き、回転RMSEは33.80°。隠れた部分を含む投影輪郭の平均IoUは0.445 |
| 400から456まで8コマおき | 8 | 復元できず | 初期コマに続く姿勢が得られず、復元できず |

第1区間で模型は初期姿勢から最後まで約64°回っていますが、復元結果の回転量は約6°でした。輪郭の重なりや追跡コマ数だけでは、歩行器の脚が何度回ったかを評価できません。実際の歩行器の角度誤差は未評価です。

BOP形式で展開したデータセットがあれば、次のコマンドで第1区間を再現できます。最初のコマンドが表示する復元コマンドを実行してから採点します。第2区間は両コマンドの範囲を `--first 400 --last 456` に変えて実行します。

```bash
python scripts/video_bop_clip.py prepare --scene-dir /data/tudl/test/000001 --first 0 --last 88 --output-dir /tmp/tudl-first
python scripts/video_bop_clip.py score --scene-dir /data/tudl/test/000001 --first 0 --last 88 --reconstruction /tmp/tudl-first/scene.json
```
