#!/usr/bin/env python3
"""Generate a sample audio file for the README demo."""

from pathlib import Path

SAMPLE_TEXT = (
    "Welcome to Talking Snake! Upload any PDF or paste a URL, " "and I will read it aloud for you."
)


def main() -> None:
    """Generate the sample audio file."""
    from talking_snake.tts import QwenTTSEngine

    print("Loading TTS engine...")
    engine = QwenTTSEngine()

    print(f"Generating audio for sample text ({len(SAMPLE_TEXT)} characters)...")
    output_path = Path(__file__).parent.parent / "src" / "talking_snake" / "static" / "sample.wav"

    # Collect all audio chunks
    audio_bytes = b""
    for chunk in engine.synthesize(SAMPLE_TEXT):
        audio_bytes += chunk

    # Write to file
    output_path.write_bytes(audio_bytes)
    print(f"Sample audio saved to: {output_path}")
    print(f"File size: {len(audio_bytes) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
