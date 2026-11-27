# MASt3R Photo to 3D

同じ被写体を異なる位置から撮った写真 **2〜10枚** を、NAVERのMASt3Rで一括処理する小さなアプリです。復元したカラー点群は、ブラウザで回転・拡大して確認し、PLYで保存できます。入力形式はJPEGとPNGです。

**[GitHubでアプリを起動](https://codespaces.new/temesotejam/mast3r-photo-to-3d?quickstart=1)** · [バッチ復元の実行履歴](https://github.com/temesotejam/mast3r-photo-to-3d/actions/workflows/reconstruct.yml)

## 自分の写真を使う

1. 上の「GitHubでアプリを起動」を開き、GitHubにサインインして、このリポジトリのCodespaceを作成または再開します。
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
