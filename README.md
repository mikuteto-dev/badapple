# badapple

Bad Appleを学習済みアテンションマップで再生する。
既存の有名実装（brayevalerien/bad-apple-but-its-gpt2）は凍結GPT-2 XLの入力埋め込みをフレームごとに独立最適化するが、これは重み込みの学習を1回の訓練で済ませ、チェックポイントから全フレームを推理して再生する。causal maskを捨てているため256×256全画素を使える。

## 実行

```sh
uv sync
uv run python badapple.py download          # yt-dlp + ffmpeg: 15fps, 256x256 gray
uv run python badapple.py train --steps 12000 --batch 512 --snapshots 10
uv run python badapple.py render
```

- `train`: 3286フレームの行正規化ログをターゲットに、入力埋め込みx（フレームごと）とQ/K重みをAdamで同時学習。lossはsoftmax前のlogit空間MSE（softmax後だと勾配消失する）。
- チェックポイントごとに`out/progress_XXXpct_audio.mp4`（訓練進捗での再生）を出す。
- `render`: 最終モデルで全フレーム再生し`out/final_audio.mp4`をmuxする。

## 要件

Apple Silicon（MPS）またはCUDA、16GB RAM、ffmpeg、yt-dlp。MPS環境では稀にプロセスクラッシュすることがある（再実行で継続）。
