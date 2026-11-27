# 単眼動画から動く部品を復元する

同じ脚が見え隠れする動画では、脚がよく見えた時刻に写った模様を、後で脚が重なった時刻にも利用できます。このアプリは、最初のコマで囲んだ部品をそれぞれ剛体として扱い、観測できた面を部品座標系に蓄積します。

## 映像から向きを推定する手がかり

| 読んだ論文 | この復元との関係 |
|---|---|
| Wen ほか、BundleSDF（CVPR 2023） | 遮蔽後に再発見した物体を、保存しておいたキーフレームの特徴と照合します。原論文はRGB-Dカメラを使います。本実装はRGB動画の前後のコマから参照候補を取り、奥行きの初期値をMASt3Rで求めます。[本文](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf)・[補足 §1](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wen_BundleSDF_Neural_6-DoF_CVPR_2023_supplemental.pdf) |
| Engel ほか、Direct Sparse Odometry（IEEE TPAMI 2018） | 位置合わせに画素の明るさを直接使います。本実装では複数コマで確認できた平らな面の模様を候補姿勢で元映像に投影し、色のずれを減らします。計算から取り分けておいた別の画像領域でも照合します。[本文 §2](https://cvai.cit.tum.de/_media/spezial/bib/engel_et_al_pami2018.pdf) |
| Schönberger ほか、Pixelwise View Selection for Unstructured Multi-View Stereo（ECCV 2016） | よく似た画像でも、同じ方向から見ていると奥行きを決める視差が足りません。論文は画素の見た目、三角測量の角度、再投影の整合性を使って参照画像を選びます。[本文 §4](https://demuc.de/papers/schoenberger2016mvs.pdf) |
| Yang ほか、LASR（CVPR 2021） | 1本のRGB動画に対し、推定した立体から生成した輪郭・画像上の動き・画素値を元映像と照合して形状と関節運動を求めます。本アプリの輪郭評価は、推定した部品を元の時刻に投影して正解輪郭と比べるものです。[本文 §3](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_LASR_Learning_Articulated_Shape_Reconstruction_From_a_Monocular_Video_CVPR_2021_paper.pdf) |

## 正解が分かる動画での評価

- **回転誤差**は初期姿勢から各時刻までの3D回転について、正解と推定との間の角度を度数で表し、二乗平均平方根を求めます。姿勢を補間した時刻も集計し、観測・補間・追跡不可の件数を併記します。
- **輪郭のIoU**は、各部品を元のカメラに投影した点群と正解輪郭が重なる面積を、両者を合わせた面積で割った値です。**厚み／長さ**は点群を初期姿勢の既知の部品軸に移し、各方向の1〜99パーセンタイルの幅で比較します。
- **見えた表面の精度と再現率**には、入力のいずれかの時刻で実際に見えた正解の表面だけを使います。単眼の原点とスケールの違いを取り除くため、部品ごとに移動と一様倍率を合わせ、回転は固定します。部品長の2%以内に正解面がある復元点の割合を精度、逆に復元点がある正解点の割合を再現率とし、調和平均をFスコアとします。実寸や部品間の距離はこの指標の対象外です。[距離から精度・再現率を計算する手法の原典](https://www.tanksandtemples.org/tutorial/)

正解の姿勢、奥行き、部品ラベルは評価にだけ使います。GitHub Actionsの公開デモでは同じ動画・初期指定が以前に公開されている場合、前後の数値を同じ定義で比較できます。[デモを再生](https://temesotejam.github.io/mast3r-photo-to-3d/video.html)・[各コマの誤差と3D表面の評価](https://temesotejam.github.io/mast3r-photo-to-3d/video-example/quality.json)
