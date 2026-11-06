# badapple

Bad Appleを学習済みattention headで再生する。既存の有名実装（brayevalerien/bad-apple-but-its-gpt2）は凍結GPT-2 XLの入力埋め込みをフレームごとに独立最適化するが、これは1回の訓練で全フレームを重みに記憶させ、チェックポイント時点のモデルで全フレームを推理して再生する。causal maskを捨てているため256×256全画素を使える。

## 2つの再生モード

- **アテンションマップ**: `logits = (xWq)(xWk)ᵀ/√d` の256×256行列をそのまま表示
- **生パラメータ**: 入力埋め込みxの前半64次元をフレームを4:1に潰した像に釘付け（aux損失）。**復号なし**で重みを並べるだけで動画になる

スナップショット動画は左＝生パラメータ／右＝アテンションの並列表示。

## 音声

曲をスペクトログラム（時間×対数周波数帯）にして同じhead構造の第2ヘッドに暗記させ、Griffin-Limで位相を推定して逆合成する。元音源は使わない。時間あたりの音圧はxへの線形読み出し（aux損失）で学習する。

## 実行

```sh
uv sync
uv run python badapple.py download      # yt-dlp + ffmpeg: 15fps, 256x256 gray
uv run python badapple.py train --steps 9000 --batch 512 --dim 128 --aux 2.0 --snapshots 10
uv run python badapple.py train --resume out/final.pt --steps 3000 --aux 10 --snapshots 3  # param viewのピン留め
uv run python badapple.py song          # 音声ヘッド学習 + 逆合成 + mux
uv run python badapple.py render        # attention再生 final.mp4
uv run python badapple.py render --params
```

## 設計の要点

- ターゲット: フレームを行正規化→log→行centering（softmaxの行和=1とシフト不変性に整合）
- loss: **softmax前のlogit空間MSE**（softmax後は勾配が潰れる）
- xの前半64次元＝映像（aux専有）、後半64次元＝attentionの自由度。rank 128の表示でv1（rank 64）より良い画質
- Q/K共有は全フレームの目標を動かすのでlrを1/10に
- lr再起動（warm restart）は共有Q/Kでは悪化する。詰まったフレームへの対処はauxの直接教師が担う

## 実測（Apple Silicon 16GB, MPS）

- 訓練 9000+3000ステップ ≈ 55分、フレーム相関 mean **0.594**（rank 64・auxなしのv1は0.584）
- スナップショット動画の左半分（生重み）とターゲットの相関 **0.99**
- 音声ヘッド D=256フルランク、log-MSE 0.004、元スペクトルとのlog-mag相関 0.69
- 表示パイプラインはGPU一括化（z-score→percentile→separable blur→LUT、12.9→0.35ms/フレーム）
- 16GB環境ではMPSメモリでjetsam Killが起きるため、ターゲットはCPU置き＋snapshotごとにcheckpoint保存

`bench/` に最適化の測定記録（candidates.md、生データ、プロファイル）がある。
