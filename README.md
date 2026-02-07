# Talking Snake

<img src="https://raw.githubusercontent.com/LucaCappelletti94/talking-snake/main/talking_snake.png" alt="Talking Snake" width="400">

[![PyPI](https://img.shields.io/pypi/v/talking-snake.svg)](https://pypi.org/project/talking-snake/)
[![CI](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LucaCappelletti94/talking-snake/branch/main/graph/badge.svg)](https://codecov.io/gh/LucaCappelletti94/talking-snake)

PDF and web page to speech using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice). Upload a document or URL, get it read aloud with 9 natural voices across English, Chinese, Japanese, and Korean. Audio streams progressively while generation continues.

## Deploy Your Own

[![Deploy on Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-lg.svg)](https://huggingface.co/spaces/LucaCappelletti94/talking-snake?duplicate=true)

Click the button above to deploy your own GPU-powered instance. You'll be prompted to create a Hugging Face account and select hardware (L4 or A100 recommended for speed, ~$0.80-$4/hr).

## Run Locally

Requires Python 3.11+, NVIDIA GPU (~6GB VRAM), and [SoX](https://sourceforge.net/projects/sox/) (`apt install sox libsox-dev`). The GPU will be automatically freed if the app is idle for 5+ minutes. It can also run on CPU (no GPU, but much slower).

```bash
uv sync && uv run --no-sync talking-snake --port 8888  # Open http://localhost:8888
```

### Flash Attention (Optional, ~2x faster)

Flash Attention requires matching your CUDA driver version. Check yours with `nvidia-smi` (top right shows "CUDA Version").

1. Find a prebuilt wheel at [flashattn.dev](https://flashattn.dev/#finder) matching your:
   - CUDA version (e.g., cu130 for CUDA 13.0)
   - PyTorch version (e.g., torch2.10)
   - Python version (e.g., cp312 for Python 3.12)

2. Install matching torch, torchaudio, and flash-attn:

   ```bash
   # Example for CUDA 13.0 + PyTorch 2.10 + Python 3.12
   uv pip install torch==2.10.0+cu130 torchaudio==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
   uv pip install <flash-attn-wheel-url>
   ```

3. Run with `--no-sync` to prevent uv from removing the manually installed packages:

   ```bash
   uv run --no-sync talking-snake --port 8888
   ```

[▶️ Listen to a sample](https://github.com/LucaCappelletti94/talking-snake/raw/main/src/talking_snake/static/sample.wav)

The website looks like this:

<img src="https://raw.githubusercontent.com/LucaCappelletti94/talking-snake/main/landing.png" alt="Upload interface" width="400">
<img src="https://raw.githubusercontent.com/LucaCappelletti94/talking-snake/main/rendering.png" alt="Audio playback with progress" width="400">

## License

This project is licensed under the [MIT License](LICENSE). Dependencies and third-party components (e.g., Qwen3-TTS, SoX) are subject to their own licenses.
