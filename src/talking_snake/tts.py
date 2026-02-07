"""TTS engine wrapper for Qwen3-TTS."""

from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class TTSEngineProtocol(ABC):
    """Protocol for TTS engines, enabling dependency injection and mocking."""

    @abstractmethod
    def synthesize(self, text: str) -> Iterator[bytes]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.

        Yields:
            WAV audio data chunks.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of generated audio."""
        ...

    @property
    def batch_size(self) -> int:
        """Return the batch size for parallel processing (default: 1)."""
        return 1


# Professional narration style prompt
# This instructs the model to read with clear, authoritative delivery
PROFESSIONAL_STYLE = (
    "Read this as a professional narrator with clear enunciation, "
    "measured pacing, and an authoritative yet warm tone. "
    "Speak naturally as if presenting an audiobook or documentary. "
    "Avoid sounding robotic or monotone. Emphasize key points and maintain a steady rhythm. "
    "Use appropriate intonation to convey meaning and keep the listener engaged. "
    "This is not casual conversation, but a polished narration style. "
    "Use proper diction, read correctly acronyms, and pronounce all words clearly."
)

# Language to default voice mapping
LANGUAGE_VOICES: dict[str, str] = {
    "english": "Ryan",
    "chinese": "Vivian",
    "japanese": "Ono_Anna",
    "korean": "Sohee",
}

# Default chunk size for streaming
# Larger chunks = more stable voice, fewer artifacts at boundaries
# Smaller chunks = faster first audio but potential voice instability
# 1200 chars provides good balance for natural speech flow
DEFAULT_CHUNK_SIZE = 1200


class QwenTTSEngine(TTSEngineProtocol):
    """TTS engine using Qwen3-TTS model."""

    # Available voices for CustomVoice model:
    # Chinese: Vivian, Serena, Uncle_Fu, Dylan (Beijing), Eric (Sichuan)
    # English: Ryan, Aiden
    # Japanese: Ono_Anna
    # Korean: Sohee
    AVAILABLE_VOICES = [
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    ]

    def __init__(
        self,
        voice: str | None = None,
        language: str = "english",
        device: str = "cuda",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ) -> None:
        """Initialize the TTS engine.

        Args:
            voice: Voice name to use for synthesis. If None, uses default for language.
                   Available voices:
                   Chinese: Vivian, Serena, Uncle_Fu, Dylan, Eric
                   English: Ryan, Aiden
                   Japanese: Ono_Anna
                   Korean: Sohee
            language: Language for TTS. One of: english, chinese, japanese, korean.
                      Sets default voice if voice is None.
            device: Device to run the model on ('cuda' or 'cpu').
            chunk_size: Maximum characters per chunk (smaller = faster streaming start).
            model_name: HuggingFace model identifier.
        """
        import logging
        import warnings

        import torch
        from qwen_tts import Qwen3TTSModel

        # Suppress the pad_token_id warning from transformers
        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")

        self.language = language.lower()
        self.voice = voice or LANGUAGE_VOICES.get(self.language, "Ryan")
        self.device = device
        self.chunk_size = chunk_size
        self._sample_rate = 24000
        self._batch_size = 1  # Will be calculated after model loads

        # Determine dtype based on device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Try to use flash attention on CUDA
        attn_impl = "flash_attention_2" if device == "cuda" else "eager"

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception:
            # Fallback without flash attention
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device,
                dtype=dtype,
            )

        # Calculate optimal batch size based on available VRAM
        if device == "cuda":
            self._batch_size = self._calculate_batch_size()
            print(f"   Batch size: {self._batch_size} (based on available VRAM)")

    def _calculate_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory.

        Returns:
            Recommended batch size for parallel chunk processing.
        """
        import torch

        if not torch.cuda.is_available():
            return 1

        try:
            # Get GPU memory info
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)

            # Available memory (conservative estimate)
            available = gpu_mem - max(allocated, reserved)

            # Model uses ~6GB, each batch item needs ~2-3GB for generation
            # Use conservative 3GB per batch item estimate
            mem_per_batch = 3 * 1024 * 1024 * 1024  # 3GB

            # Calculate batch size, minimum 1, cap at 8
            batch_size = max(1, min(8, int(available / mem_per_batch)))

            return batch_size
        except Exception:
            return 1

    @property
    def sample_rate(self) -> int:
        """Return the sample rate of generated audio."""
        return self._sample_rate

    @property
    def batch_size(self) -> int:
        """Return the current batch size."""
        return self._batch_size

    def synthesize(self, text: str) -> Iterator[bytes]:
        """Synthesize text to WAV audio using batched GPU inference.

        Args:
            text: Text to synthesize.

        Yields:
            WAV audio data chunks.
        """
        if not text.strip():
            return

        # Split text into chunks for streaming
        chunks = self._split_text(text)

        # First chunk includes WAV header
        first_chunk = True

        # Process chunks in batches for GPU efficiency
        batch_size = self._batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Filter empty chunks
            batch = [c for c in batch if c.strip()]
            if not batch:
                continue

            # Always use batched call for consistent GPU memory allocation
            # Use professional narration style for clear, authoritative delivery
            batch_instruct = (
                [PROFESSIONAL_STYLE] * len(batch) if len(batch) > 1 else PROFESSIONAL_STYLE
            )
            audios, sr = self.model.generate_custom_voice(
                text=batch if len(batch) > 1 else batch[0],
                speaker=[self.voice] * len(batch) if len(batch) > 1 else self.voice,
                instruct=batch_instruct,
                # Use lower temperature for more stable, consistent voice
                temperature=0.7,
                repetition_penalty=1.1,
            )

            # Ensure audios is a list for consistent iteration
            if len(batch) == 1:
                audios = [audios]

            # Yield each audio chunk in order
            for audio in audios:
                wav_bytes = self._audio_to_wav(audio, sr, include_header=first_chunk)
                first_chunk = False
                yield wav_bytes

    def _split_text(self, text: str, max_chars: int | None = None) -> list[str]:
        """Split text into chunks suitable for TTS.

        Splits on sentence boundaries when possible.

        Args:
            text: Text to split.
            max_chars: Maximum characters per chunk. Uses self.chunk_size if None.

        Returns:
            List of text chunks.
        """
        import re

        if max_chars is None:
            max_chars = self.chunk_size

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _audio_to_wav(
        self,
        audio: npt.NDArray[np.float32] | list[float],
        sample_rate: int,
        include_header: bool = True,
    ) -> bytes:
        """Convert audio array to WAV bytes.

        Args:
            audio: Audio data as numpy array or list.
            sample_rate: Sample rate of the audio.
            include_header: Whether to include WAV header.

        Returns:
            WAV audio data as bytes.
        """
        import numpy as np

        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Normalize and convert to 16-bit PCM
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        if include_header:
            # Write full WAV file
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            result: bytes = buffer.getvalue()
            return result
        else:
            # Return raw PCM data
            pcm_data: bytes = audio_int16.tobytes()
            return pcm_data


class MockTTSEngine(TTSEngineProtocol):
    """Mock TTS engine for testing."""

    def __init__(self, sample_rate: int = 24000) -> None:
        """Initialize the mock TTS engine.

        Args:
            sample_rate: Sample rate for generated audio.
        """
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        """Return the sample rate of generated audio."""
        return self._sample_rate

    def synthesize(self, text: str) -> Iterator[bytes]:
        """Generate silent WAV audio for testing.

        Args:
            text: Text to synthesize (used to determine duration).

        Yields:
            WAV audio data with silence.
        """
        if not text.strip():
            return

        # Generate ~0.1 seconds of silence per word
        words = len(text.split())
        duration_samples = int(self._sample_rate * 0.1 * max(1, words))

        # Create silent audio
        silence = b"\x00\x00" * duration_samples

        # Write WAV header + silence
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(silence)

        yield buffer.getvalue()
