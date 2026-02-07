"""Tests for TTS engine."""

from __future__ import annotations

import wave
from io import BytesIO
from typing import TYPE_CHECKING

import pytest

from talking_snake.tts import MockTTSEngine, TTSEngineProtocol

if TYPE_CHECKING:
    from talking_snake.tts import QwenTTSEngine


class TestMockTTSEngine:
    """Tests for the mock TTS engine."""

    def test_implements_protocol(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that MockTTSEngine implements TTSEngineProtocol."""
        assert isinstance(mock_tts_engine, TTSEngineProtocol)

    def test_sample_rate(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that sample rate is accessible."""
        assert mock_tts_engine.sample_rate == 24000

    def test_synthesize_returns_iterator(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that synthesize returns an iterator."""
        result = mock_tts_engine.synthesize("Hello world")
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_synthesize_yields_bytes(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that synthesize yields bytes."""
        for chunk in mock_tts_engine.synthesize("Hello world"):
            assert isinstance(chunk, bytes)

    def test_synthesize_produces_valid_wav(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that synthesize produces valid WAV data."""
        chunks = list(mock_tts_engine.synthesize("Hello world"))
        assert len(chunks) > 0

        # Combine all chunks
        wav_data = b"".join(chunks)

        # Parse as WAV
        wav_file = wave.open(BytesIO(wav_data), "rb")
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2  # 16-bit
        assert wav_file.getframerate() == 24000
        wav_file.close()

    def test_synthesize_empty_text(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that empty text produces no output."""
        chunks = list(mock_tts_engine.synthesize(""))
        assert len(chunks) == 0

    def test_synthesize_whitespace_only(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that whitespace-only text produces no output."""
        chunks = list(mock_tts_engine.synthesize("   \n\t  "))
        assert len(chunks) == 0

    def test_duration_scales_with_text(self, mock_tts_engine: MockTTSEngine) -> None:
        """Test that audio duration scales with text length."""
        short_chunks = list(mock_tts_engine.synthesize("Hi"))
        long_chunks = list(mock_tts_engine.synthesize("Hello world this is a longer sentence"))

        short_wav = wave.open(BytesIO(b"".join(short_chunks)), "rb")
        long_wav = wave.open(BytesIO(b"".join(long_chunks)), "rb")

        short_frames = short_wav.getnframes()
        long_frames = long_wav.getnframes()

        short_wav.close()
        long_wav.close()

        assert long_frames > short_frames


class TestMockTTSEngineCustomSampleRate:
    """Tests for MockTTSEngine with custom sample rate."""

    def test_custom_sample_rate(self) -> None:
        """Test that custom sample rate is used."""
        engine = MockTTSEngine(sample_rate=16000)
        assert engine.sample_rate == 16000

        chunks = list(engine.synthesize("Hello"))
        wav_file = wave.open(BytesIO(b"".join(chunks)), "rb")
        assert wav_file.getframerate() == 16000
        wav_file.close()


class TestAudioToWavConversion:
    """Tests for audio conversion utilities."""

    def test_audio_to_wav_with_numpy_array(self) -> None:
        """Test _audio_to_wav handles numpy arrays correctly."""
        import numpy as np

        from talking_snake.tts import QwenTTSEngine

        # Create a simple audio array
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

        # Access the method via the class (we can't instantiate without GPU)
        # So we test the logic directly
        engine = type(
            "TestEngine",
            (),
            {"_audio_to_wav": QwenTTSEngine._audio_to_wav},
        )()

        result = engine._audio_to_wav(audio, 24000, include_header=True)

        # Should produce valid WAV
        assert result[:4] == b"RIFF"
        assert b"WAVE" in result[:12]

    def test_audio_to_wav_with_list(self) -> None:
        """Test _audio_to_wav handles Python lists correctly (regression test)."""
        from talking_snake.tts import QwenTTSEngine

        # This is what qwen-tts sometimes returns
        audio = [0.0, 0.5, -0.5, 1.0, -1.0]

        engine = type(
            "TestEngine",
            (),
            {"_audio_to_wav": QwenTTSEngine._audio_to_wav},
        )()

        result = engine._audio_to_wav(audio, 24000, include_header=True)

        # Should produce valid WAV
        assert result[:4] == b"RIFF"
        assert b"WAVE" in result[:12]

    def test_audio_to_wav_with_2d_array(self) -> None:
        """Test _audio_to_wav flattens 2D arrays."""
        import numpy as np

        from talking_snake.tts import QwenTTSEngine

        # 2D array (stereo or batch)
        audio = np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float32)

        engine = type(
            "TestEngine",
            (),
            {"_audio_to_wav": QwenTTSEngine._audio_to_wav},
        )()

        result = engine._audio_to_wav(audio, 24000, include_header=True)

        # Should produce valid WAV
        wav_file = wave.open(BytesIO(result), "rb")
        assert wav_file.getnchannels() == 1
        wav_file.close()

    def test_audio_to_wav_without_header(self) -> None:
        """Test _audio_to_wav can produce raw PCM data."""
        import numpy as np

        from talking_snake.tts import QwenTTSEngine

        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)

        engine = type(
            "TestEngine",
            (),
            {"_audio_to_wav": QwenTTSEngine._audio_to_wav},
        )()

        result = engine._audio_to_wav(audio, 24000, include_header=False)

        # Should be raw PCM (no RIFF header)
        assert result[:4] != b"RIFF"
        # 3 samples * 2 bytes per sample = 6 bytes
        assert len(result) == 6

    def test_audio_to_wav_clips_values(self) -> None:
        """Test _audio_to_wav clips values outside [-1, 1]."""
        import numpy as np

        from talking_snake.tts import QwenTTSEngine

        # Values outside valid range
        audio = np.array([-2.0, 2.0], dtype=np.float32)

        engine = type(
            "TestEngine",
            (),
            {"_audio_to_wav": QwenTTSEngine._audio_to_wav},
        )()

        result = engine._audio_to_wav(audio, 24000, include_header=True)

        # Should produce valid WAV (values clipped)
        wav_file = wave.open(BytesIO(result), "rb")
        frames = wav_file.readframes(2)
        wav_file.close()

        # Check that values are clipped to 16-bit min/max
        import struct

        samples = struct.unpack("<hh", frames)
        assert samples[0] == -32767  # Clipped from -2.0
        assert samples[1] == 32767  # Clipped from 2.0


@pytest.mark.slow
class TestQwenTTSEngine:
    """Tests for the real Qwen TTS engine.

    These tests require the actual model and GPU.
    Run with: pytest --run-slow
    """

    @pytest.fixture
    def qwen_engine(self) -> QwenTTSEngine:
        """Load the Qwen TTS engine."""
        from talking_snake.tts import QwenTTSEngine

        return QwenTTSEngine(voice="Chelsie", device="cuda")

    def test_implements_protocol(self, qwen_engine: QwenTTSEngine) -> None:
        """Test that QwenTTSEngine implements TTSEngineProtocol."""

        assert isinstance(qwen_engine, TTSEngineProtocol)

    def test_synthesize_produces_audio(self, qwen_engine: QwenTTSEngine) -> None:
        """Test that synthesize produces actual audio."""
        chunks = list(qwen_engine.synthesize("Hello, this is a test."))
        assert len(chunks) > 0

        wav_data = b"".join(chunks)
        assert len(wav_data) > 1000  # Should have substantial audio data

    def test_synthesize_long_text(self, qwen_engine: QwenTTSEngine) -> None:
        """Test synthesis of longer text with chunking."""
        long_text = """
        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience. This technology has
        revolutionized many fields including natural language processing,
        computer vision, and speech recognition.
        """
        chunks = list(qwen_engine.synthesize(long_text))
        assert len(chunks) > 0
