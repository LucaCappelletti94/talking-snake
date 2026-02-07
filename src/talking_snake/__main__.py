"""CLI entry point for the Reader server."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Main entry point for the Reader CLI.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog="reader",
        description="PDF-to-Speech web server - listen to any content",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice name for TTS. Options: Vivian, Serena, Uncle_Fu, Dylan, Eric, "
        "Ryan, Aiden, Ono_Anna, Sohee (default: auto based on language)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "chinese", "japanese", "korean"],
        help="Language for TTS (default: english). Sets default voice if --voice not specified.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run the TTS model on (default: auto, detects GPU)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Auto-detect device if set to 'auto'
    device = args.device
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                print("🎮 GPU detected, using CUDA")
            else:
                device = "cpu"
                print("💻 No GPU detected, using CPU (slower but works!)")
        except ImportError:
            device = "cpu"
            print("💻 PyTorch not available for detection, using CPU")

    print("🚀 Starting Reader server...")
    print(f"   Language: {args.language}")
    print(f"   Voice:    {args.voice or 'auto'}")
    print(f"   Device:   {device}")
    print(f"   URL:      http://{args.host}:{args.port}")
    print()

    # Import here to avoid slow startup for --help
    import uvicorn

    from talking_snake.app import create_app
    from talking_snake.tts import QwenTTSEngine

    # Initialize TTS engine
    print("📦 Loading TTS model (this may take a moment)...")
    try:
        tts_engine = QwenTTSEngine(
            voice=args.voice,
            language=args.language,
            device=device,
        )
    except Exception as e:
        print(f"❌ Failed to load TTS model: {e}", file=sys.stderr)
        return 1

    print("✅ TTS model loaded!")

    # Run calibration to get accurate time estimates
    print("⏱️  Calibrating speech timing...")
    try:
        tts_engine.calibrate()
    except Exception as e:
        print(f"⚠️  Calibration failed (using defaults): {e}")
    print()

    # Create app with engine
    app = create_app(tts_engine=tts_engine)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
