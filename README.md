# MASt3R Photo to 3D

同じ被写体を異なる位置から撮った写真 **2〜10枚** を、NAVERのMASt3Rで一括処理する小さなアプリです。復元したカラー点群は、ブラウザで回転・拡大して確認し、PLYで保存できます。入力形式はJPEGとPNGです。

**[GitHub Pagesで操作画面を開く](https://temesotejam.github.io/mast3r-photo-to-3d/)** · [Codespacesで復元する](https://codespaces.new/temesotejam/mast3r-photo-to-3d?quickstart=1) · [バッチ復元の実行履歴](https://github.com/temesotejam/mast3r-photo-to-3d/actions/workflows/reconstruct.yml)

## 写真4枚から復元した点群

[中庭の点群をブラウザで回して見る](https://temesotejam.github.io/mast3r-photo-to-3d/?sample=1) · [カラーPLYをダウンロード](https://temesotejam.github.io/mast3r-photo-to-3d/sample.ply) · [入力写真と出典](docs/examples/courtyard/)

公開データセットETH3Dの中庭を異なる位置から撮った4枚を、アプリの写真受付APIに送って生成した実例です。この表示にはサインインも計算環境の起動も必要ありません。

## 動く物体を動画から3Dで再生する

**[元動画と復元結果を並べて再生](https://temesotejam.github.io/mast3r-photo-to-3d/video.html)** · [動画の復元履歴](https://github.com/temesotejam/mast3r-photo-to-3d/actions/workflows/video.yml)

動画モードでは、第1フレームで指定した部品ごとに一つのカラー点群を作成します。部品が回って新しい面が見えると、その観測を同じ部品座標系に登録します。再生中は、動画全体で蓄積した点群を各時刻の姿勢に配置します。ビューをドラッグして視点を変え、スライダーで好きな時刻に移動できます。

1. 自分の動画を処理するには、上のCodespacesリンクから作業環境を起動し、ポート7861のアプリで **「動画から4D復元」** を開きます。ローカルの起動方法は写真モードと共通です。
2. 30秒以内のMP4・MOV・WebMを選びます。第1フレームで見えている胴体や脚など、**形が変わらず一体で動く部分**を個別に四角く囲み、名前を付けます。1回で1〜6部品を指定できます。対象の表面に模様があり、隣の部品や背景を囲みに含めすぎない動画が適しています。
3. 固定撮影なら固定カメラを選びます。移動カメラでは画面内の静止背景を使ってカメラの動きを推定するため、十分な広さと模様のある背景が必要です。復元後は時系列を含むJSONを保存し、後日ビューアで再度開けます。

フレームは動画全体から最大24枚、最大約4 Hzで抽出します。MASt3Rの推定深度と画像上の対応点から各部品の位置・回転を求めます。見失ってから再び検出した区間は前後の姿勢で補間し、補間中は形状を追加しません。片側の時刻しか観測できなかった区間は姿勢不明として扱います。単眼の推定深度と部品ごとの剛体運動を使うため、点の距離・角度や接地時刻を計測値として使う場合には別途検証が必要です。動画を通じて一度も見えなかった面を作成する機能や、関節軸を拘束する機能はありません。

公開デモは2部品が前後に重なって再び離れる合成動画です。推定器に与える情報は通常のRGB動画と最初の囲みだけです。シミュレーションの正解位置と可視率は復元後の動作検証に使います。

GitHub Actionsから自分の素材を処理するには、動画と囲みJSONのパスを [動画ワークフロー](https://github.com/temesotejam/mast3r-photo-to-3d/actions/workflows/video.yml) に指定します。**公開リポジトリにコミットした素材は公開されます。** 入力JSONの `box` は、先頭フレーム上の左・上・右・下を0〜1に正規化した値です。

```json
[{"name":"胴体","box":[0.35,0.14,0.62,0.43]},{"name":"左脚","box":[0.37,0.46,0.48,0.89]}]
```

```sh
.venv/bin/python reconstruct_video.py --video input.mp4 --parts parts.json --camera fixed --output results/scene.json
```

## 自分の写真を使う

1. 上の「Codespacesで復元する」を開き、GitHubにサインインして、このリポジトリのCodespaceを作成または再開します。
2. 初回はPython環境を自動で準備します。作業画面の **Ports（ポート）** に `7861` が現れたら、その行の地球アイコンからアプリを開きます。
3. 同じ被写体の写真を2〜10枚選び、**「3Dを復元」** を押します。初回の復元ではモデルもダウンロードします。画面に完了と表示されるまでCodespaceを起動したままにします。
4. 点群をドラッグして回し、ホイールで拡大します。**「PLYを保存」** で手元に保存できます。利用後は [Codespaces一覧](https://github.com/codespaces) から作業環境を停止できます。

写真は起動中の自分のCodespaceに送られ、Gitの履歴には入りません。計算にはそのアカウントのCodespaces利用枠を使います。最初の撮影では、机に被写体を固定し、カメラを左斜めから右斜めまで少しずつ移動して4〜6枚撮ると試しやすいです。隣り合う写真に同じ模様や輪郭を十分に残してください。

## GitHub Actionsでまとめて復元する

1. [Actionsの復元画面](https://github.com/temesotejam/mast3r-photo-to-3d/actions/workflows/reconstruct.yml) で **Run workflow** を選びます。サンプルを使う設定では同梱の公開サンプル写真で試せます。
2. 自分の写真で実行する場合は、あらかじめリポジトリの `photos/` など同じフォルダーに写真を追加し、サンプル指定を外して、そのフォルダー名を入力します。**このリポジトリは公開されているため、コミットした写真も公開されます。** 写真を公開したくない場合はCodespacesまたは下記のローカル実行が使えます。
3. 実行が完了したら、その実行の **Artifacts → mast3r-point-cloud** をダウンロードします。ZIP内の `cloud.ply` を操作画面の **「PLYを開く」** で表示します。

## 自分のPCで動かす

GitとPython 3.11を用意して、このリポジトリをクローンします。以下はLinuxでのCPU版の例です。

```sh
git clone https://github.com/temesotejam/mast3r-photo-to-3d.git
cd mast3r-photo-to-3d
python3.11 -m venv .venv
.venv/bin/python scripts/setup_cpu.py
.venv/bin/python server.py
```

WindowsのPowerShellでは、クローン先で次を実行します。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe scripts/setup_cpu.py
.\.venv\Scripts\python.exe server.py
```

ブラウザで [ローカルの操作画面](http://127.0.0.1:7861/) を開きます。写真は自分のPCで処理されます。モデルと依存ライブラリの初回取得にはインターネット接続が必要です。

フォルダーの写真をコマンドで処理する場合は、同じ仮想環境で以下を実行します。

```sh
.venv/bin/python reconstruct.py --input photos --output results/cloud.ply --device cpu
```

## モデル

写真から推定した3次元点と元画像の色をPLYの各頂点に保存します。寸法計測に使う場合は、既知寸法の物体を一緒に撮ってスケールと精度を確認してください。

上流の実装と重みの利用にはNAVERおよび学習データの利用条件が適用されます。詳細は [MASt3R公式リポジトリ](https://github.com/naver/mast3r)、[LICENSE](https://github.com/naver/mast3r/blob/main/LICENSE)、[CHECKPOINTS_NOTICE](https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE) を参照してください。
