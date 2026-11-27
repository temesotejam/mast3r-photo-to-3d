# 動いている部品を1本の動画から復元する方法

[同じ動画を方式ごとに再生する](https://temesotejam.github.io/mast3r-photo-to-3d/video.html) · [自分の動画をGitHub Actionsで復元する](https://github.com/temesotejam/mast3r-photo-to-3d/actions/workflows/video.yml)

| 方式 | 入力から求めるもの | 試すときの注意点 |
|---|---|---|
| RGB-SfM | 模様のある点を複数コマで対応付け、同じ部品の3D点と各時刻の回転・移動を求めます。Tomasi–Kanadeの因子分解を初期値にし、元の画像への投影誤差を小さくするように推定を調整します。 | 固定カメラに対応します。表面を覆う点の数は少なく、各部品のスケールは独立です。部品間の距離や実寸の測定には使えません。[原論文](https://users.cs.duke.edu/~tomasi/papers/tomasi/tomasiIjcv92.pdf) |
| Depth Anything V2 | 室内向けのMetric Indoor Smallモデルで各RGB画像の奥行きを求め、同じ部品が見えたコマを照合して色付き点群にまとめます。 | 固定カメラでは静止背景を使い、各コマの奥行きの倍率を合わせます。メートルを予測する学習済みモデルでも、このアプリの出力の実寸は較正していません。[モデルの仕様](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) |
| MonST3R | 動きのあるシーンで学習した公式モデルの点群推定を使います。複数の画像ペアから各コマの奥行きを求めた後、このアプリの部品追跡・点群統合に渡します。 | DUSt3Rを基にした別のモデルです。この比較で使ったのは画像ペアの推論で、著者のオプティカルフローや動画全体の最適化を含むパイプラインは評価していません。[論文・公式実装](https://monst3r-project.github.io/) |

各方式とも同じ最初のコマで部品を囲みます。合成動画の正解姿勢・深度・部品ラベルは採点にだけ使います。実写はBOPのTUD-Lから人が竜の模型を動かす動画を使用し、RGBから推定する全方式に同じ初期の枠と水平画角63.71°を渡しました。回転誤差は初期姿勢を基準にした3D相対回転のRMSEです。輪郭IoUは重なった面積の割合で、疎な点群には不利な指標です。[TUD-Lと評価方法](VIDEO_ACCURACY.md)

### 最初のコマに深度画像がある場合

固定カメラで撮影した1個の剛体なら、最初のコマのRGB画像に位置合わせした深度画像と、対象が見えている部分のマスクから、その表面を色付き点群にできます。以降のコマはRGBの模様と点群の対応から姿勢を推定します。最初に写っていない面は点群に加わりません。追跡できなくなったコマは「追跡不可」として出力します。

次の例の深度PNGは画素値1が1 mm、マスクは対象が白・背景が黒です。どちらも切り抜きや縮小の前の動画と同じ解像度で、同じ画素が同じ場所を表している必要があります。固定された歪み補正済みカメラの内部行列を `K.json` に `[[fx,0,cx],[0,fy,cy],[0,0,1]]` の形で保存します。深度値はカメラの光軸方向の距離です。

```sh
python reconstruct_video_reference.py --video clip.mp4 --mask first_mask.png --depth first_depth.png --depth-scale 0.001 --intrinsics K.json --name robot-leg --output scene.json
```

メートル単位の2次元NumPy配列も `--depth first_depth.npy` で読み込めます。この場合 `--depth-scale` は省略できます。出力したJSONは動画ビューアに読み込めます。

## 関連する別の方法

**Shape of Motion**は動的シーンの自由視点再生に向いた研究です。動画全体で同じ3D点を表現し、奥行き、RGB、2Dの軌跡と合うように点の形と動きを動画ごとに最適化します。**SpatialTrackerV2**は動画中で指定した点の3D軌跡やカメラの動きを推定します。ここに掲載した独自データでの数値は、上の実行済み方式についてのものです。[Shape of Motion](https://shape-of-motion.github.io/) · [SpatialTrackerV2](https://spatialtracker.github.io/)
