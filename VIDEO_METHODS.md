# 動いている部品を1本の動画から復元する方法

[同じ動画を方式ごとに再生する](https://temesotejam.github.io/mast3r-photo-to-3d/video.html) · [自分の動画をGitHub Actionsで復元する](https://github.com/temesotejam/mast3r-photo-to-3d/actions/workflows/video.yml)

| 方式 | 入力から求めるもの | 試すときの注意点 |
|---|---|---|
| RGB-SfM | 模様のある点を複数コマで対応付け、同じ部品の3D点と各時刻の回転・移動を求めます。Tomasi–Kanadeの因子分解を初期値にし、元の画像への投影誤差を小さくするように推定を調整します。 | 固定カメラに対応します。表面を覆う点の数は少なく、各部品のスケールは独立です。部品間の距離や実寸の測定には使えません。[原論文](https://users.cs.duke.edu/~tomasi/papers/tomasi/tomasiIjcv92.pdf) |
| Depth Anything V2 | 室内向けのMetric Indoor Smallモデルで各RGB画像の奥行きを求め、同じ部品が見えたコマを照合して色付き点群にまとめます。 | 固定カメラでは静止背景を使い、各コマの奥行きの倍率を合わせます。メートルを予測する学習済みモデルでも、このアプリの出力の実寸は較正していません。[モデルの仕様](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) |
| MonST3R | 動きのあるシーンで学習した公式モデルの点群推定を使います。複数の画像ペアから各コマの奥行きを求めた後、このアプリの部品追跡・点群統合に渡します。 | DUSt3Rを基にした別のモデルです。この比較で使ったのは画像ペアの推論で、著者のオプティカルフローや動画全体の最適化を含むパイプラインは評価していません。[論文・公式実装](https://monst3r-project.github.io/) |

各方式とも同じ最初のコマで部品を囲みます。合成動画の正解姿勢・深度・部品ラベルは採点にだけ使います。実写はBOPのTUD-Lから人が竜の模型を動かす動画を使用し、RGBから推定する全方式に同じ初期の枠と水平画角63.71°を渡しました。回転誤差は初期姿勢を基準にした3D相対回転のRMSEです。輪郭IoUは重なった面積の割合で、疎な点群には不利な指標です。[TUD-Lと評価方法](VIDEO_ACCURACY.md)

## 実写の竜の模型

2区間は同じ模型を別の時刻に撮ったものです。第1区間は元データの0〜88、第2区間は400〜456を8コマおきに取り出しています。「姿勢あり」は初期コマを含みます。表に数値のある実写結果はすべてのコマで観測できました。

| 方式 | 第1区間の回転誤差（12コマ） | 第2区間の回転誤差（8コマ） | 第1区間の輪郭IoU | 第2区間の輪郭IoU |
|---|---:|---:|---:|---:|
| RGB-SfM | **4.06°** | **3.73°** | 0.277 | 0.227 |
| Depth Anything V2 | 37.58° | 14.47° | 0.448 | 0.420 |
| MonST3R（画像ペア推論） | 34.88° | 14.40° | 0.371 | 0.390 |
| MASt3R（比較基準） | 33.80° | 復元できず | 0.445 | — |
| 最初のコマに実測深度・マスク・カメラ内部行列を加えたPnP | 7.91° | 5.69° | **0.835** | **0.844** |

第1区間でRGB-SfMが追跡に使えたのは138点です。点は疎でも模様の位置関係から回転をよく捉えました。最後の方式は最初に実際に見えた面を4,296点の色付き点群として固定し、以降のRGB画像からその姿勢を推定します。どちらの点群も同じ動画に合わせて回転・拡大できます。[RGB-SfMの実写デモ](https://temesotejam.github.io/mast3r-photo-to-3d/video.html?sample=dragon-rgb-sfm) · [初期の実測点群を使った実写デモ](https://temesotejam.github.io/mast3r-photo-to-3d/video.html?sample=dragon-rgbd)

## 2本の部品が交差する合成動画

表は同じ動画16コマをGitHub ActionsのCPUで計算した結果です。回転誤差は姿勢が得られたコマで集計し、補間したコマも含みます。輪郭IoUは16コマすべてで採点し、追跡できないコマは0となります。厚み／長さは青緑色の奥の部品で比較しています。[全方式の実行記録](https://github.com/temesotejam/mast3r-photo-to-3d/actions/runs/46019413293)

| 方式 | 回転誤差（奥／手前） | 奥の部品（観測／補間／追跡不可） | 輪郭IoU（奥／手前） | 奥の厚み／長さ（正解0.132） |
|---|---:|---:|---:|---:|
| RGB-SfM | 6.60°／27.49° | 13／3／0 | 0.074／0.113 | 0.767 |
| Depth Anything V2 | 1.62°／0.27° | 11／5／0 | 0.775／0.766 | 0.521 |
| MonST3R（画像ペア推論） | 3.44°／0.27° | 8／0／8 | 0.365／0.762 | 1.273 |
| MASt3R | 1.97°／0.27° | 14／2／0 | 0.843／0.850 | 0.171 |

この動画では、Depth Anything V2は奥の回転誤差が最小でしたが、厚みを正解の約4倍に推定しています。MonST3Rは奥の部品を8コマ目まで追い、その後は再発見できませんでした。3Dの形を比較する際は、動画上の回転と、輪郭・厚みの両方を確認できます。[デモを開く](https://temesotejam.github.io/mast3r-photo-to-3d/video.html?sample=depth-anything-v2)

## 撮影条件と動きを変えた14コマ

前回のMASt3R版で奥の部品を大きく誤った動画でも比較しました。方式によって追跡できた時刻が異なるため、各方式とMASt3Rの両方で姿勢が得られたコマだけの比較を併記します。

| 方式 | 奥の部品で姿勢が得られたコマ | そのコマでの回転誤差（方式／MASt3R） | 手前の部品で姿勢が得られたコマ | そのコマでの回転誤差（方式／MASt3R） |
|---|---:|---:|---:|---:|
| RGB-SfM | 9／14 | 43.82°／63.93° | 14／14 | 15.61°／12.31° |
| Depth Anything V2 | 4／14 | 22.10°／45.68° | 10／14 | 7.46°／7.26° |
| MonST3R（画像ペア推論） | 14／14（5コマ補間） | 42.72°／66.09° | 14／14 | 32.30°／12.31° |

この動画では奥の部品の向きをどの方式でも大きく誤りました。実写の竜で得られた約4°という値を、歩行器の左右の脚が重なる区間に適用することはできません。実機の足角度は未評価です。

## 最初のコマに深度画像がある場合

固定カメラで撮影した1個の剛体なら、最初のコマのRGB画像に位置合わせした深度画像と、対象が見えている部分のマスクから、その表面を色付き点群にできます。以降のコマはRGBの模様と点群の対応から姿勢を推定します。最初に写っていない面は点群に加わりません。追跡できなくなったコマは「追跡不可」として出力します。

次の例の深度PNGは画素値1が1 mm、マスクは対象が白・背景が黒です。どちらも切り抜きや縮小の前の動画と同じ解像度で、同じ画素が同じ場所を表している必要があります。固定された歪み補正済みカメラの内部行列を `K.json` に `[[fx,0,cx],[0,fy,cy],[0,0,1]]` の形で保存します。深度値はカメラの光軸方向の距離です。

```sh
python reconstruct_video_reference.py --video clip.mp4 --mask first_mask.png --depth first_depth.png --depth-scale 0.001 --intrinsics K.json --name robot-leg --output scene.json
```

メートル単位の2次元NumPy配列も `--depth first_depth.npy` で読み込めます。この場合 `--depth-scale` は省略できます。出力したJSONは動画ビューアに読み込めます。[EPnP原論文](https://www.iri.upc.edu/people/fmoreno/Publications/2009/pdf/Lepetit_ijcv2009.pdf)

## 関連する別の方法

**Shape of Motion**は動的シーンの自由視点再生に向いた研究です。動画全体で同じ3D点を表現し、奥行き、RGB、2Dの軌跡と合うように点の形と動きを動画ごとに最適化します。**SpatialTrackerV2**は動画中で指定した点の3D軌跡やカメラの動きを推定します。ここに掲載した独自データでの数値は、上の実行済み方式についてのものです。[Shape of Motion](https://shape-of-motion.github.io/) · [SpatialTrackerV2](https://spatialtracker.github.io/)

各学習済みモデルの利用条件は配布元を確認してください。[MonST3Rのライセンス](https://github.com/Junyi42/monst3r/blob/main/LICENSE) · [Depth Anything V2のモデルカード](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) · [実写デモの出典と利用条件](docs/video-example/tudl/SOURCE.txt)
