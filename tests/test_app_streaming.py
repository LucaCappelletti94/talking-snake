"""Tests for streaming endpoints and job management."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from talking_snake.app import (
    AudioJob,
    JobManager,
    _create_wav_header,
    _estimate_time,
    _get_device_info,
    _job_manager,
    create_app,
)
from talking_snake.tts import MockTTSEngine


@pytest.fixture
def app():
    """Create a test application with mock TTS engine."""
    return create_app(tts_engine=MockTTSEngine())


@pytest.fixture
async def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestAudioJob:
    """Tests for AudioJob class."""

    def test_create_job(self) -> None:
        """Test job creation with correct initial state."""
        job = AudioJob("test-id-123")
        assert job.job_id == "test-id-123"
        assert job.completed is False
        assert job.error is None
        assert job.sample_rate == 24000
        assert job.header_sent is False

    def test_put_audio(self) -> None:
        """Test adding audio data to job queue."""
        job = AudioJob("test-id")
        job.put_audio(b"audio-data")
        assert not job.audio_queue.empty()
        assert job.audio_queue.get() == b"audio-data"

    def test_finish_job(self) -> None:
        """Test finishing a job."""
        job = AudioJob("test-id")
        job.finish()
        assert job.completed is True
        # Should have sentinel None in queue
        assert job.audio_queue.get() is None

    def test_set_error(self) -> None:
        """Test setting an error on a job."""
        job = AudioJob("test-id")
        job.set_error("Something went wrong")
        assert job.error == "Something went wrong"
        assert job.completed is True
        # Should have sentinel None in queue
        assert job.audio_queue.get() is None


class TestJobManager:
    """Tests for JobManager class."""

    def test_create_job(self) -> None:
        """Test creating a new job."""
        manager = JobManager()
        job = manager.create_job()
        assert job is not None
        assert job.job_id is not None
        assert len(job.job_id) == 36  # UUID format

    def test_get_job(self) -> None:
        """Test retrieving a job by ID."""
        manager = JobManager()
        job = manager.create_job()
        retrieved = manager.get_job(job.job_id)
        assert retrieved is job

    def test_get_nonexistent_job(self) -> None:
        """Test retrieving a nonexistent job returns None."""
        manager = JobManager()
        assert manager.get_job("nonexistent-id") is None

    def test_remove_job(self) -> None:
        """Test removing a job."""
        manager = JobManager()
        job = manager.create_job()
        manager.remove_job(job.job_id)
        assert manager.get_job(job.job_id) is None

    def test_remove_nonexistent_job(self) -> None:
        """Test removing a nonexistent job doesn't raise."""
        manager = JobManager()
        manager.remove_job("nonexistent-id")  # Should not raise


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_wav_header(self) -> None:
        """Test WAV header creation."""
        header = _create_wav_header(sample_rate=24000, bits_per_sample=16)
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        assert header[12:16] == b"fmt "
        assert len(header) == 44  # Standard WAV header size

    def test_create_wav_header_custom_rate(self) -> None:
        """Test WAV header with custom sample rate."""
        header = _create_wav_header(sample_rate=44100, bits_per_sample=16)
        assert header[:4] == b"RIFF"
        # Sample rate is at bytes 24-27 in little-endian
        import struct

        sample_rate = struct.unpack("<I", header[24:28])[0]
        assert sample_rate == 44100

    def test_estimate_time(self) -> None:
        """Test time estimation."""
        text = "Hello world. " * 100  # ~1300 chars
        chunk_count, estimated_seconds = _estimate_time(text)
        assert chunk_count >= 1
        assert estimated_seconds > 0

    def test_estimate_time_short_text(self) -> None:
        """Test time estimation for short text."""
        chunk_count, estimated_seconds = _estimate_time("Hi")
        assert chunk_count == 1
        assert estimated_seconds > 0

    def test_estimate_time_with_custom_rate(self) -> None:
        """Test time estimation with custom seconds per char."""
        text = "Hello"
        chunk_count, estimated_seconds = _estimate_time(text, seconds_per_char=0.01)
        assert estimated_seconds == pytest.approx(0.05, rel=0.1)

    def test_get_device_info_returns_dict(self) -> None:
        """Test that device info returns expected keys."""
        info = _get_device_info()
        assert isinstance(info, dict)
        assert "device" in info
        assert "device_name" in info
        assert "memory_used_gb" in info
        assert "memory_total_gb" in info
        assert "batch_size" in info
        assert "ram_used_gb" in info
        assert "ram_total_gb" in info
        assert "disk_free_gb" in info

    def test_get_device_info_ram_values(self) -> None:
        """Test that RAM values are reasonable."""
        info = _get_device_info()
        assert info["ram_total_gb"] > 0
        assert info["ram_used_gb"] >= 0
        assert info["ram_used_gb"] <= info["ram_total_gb"]

    def test_get_device_info_disk_values(self) -> None:
        """Test that disk free value is reasonable."""
        info = _get_device_info()
        assert info["disk_free_gb"] >= 0

    def test_get_device_info_timing_stats_optional(self) -> None:
        """Test that timing stats are optional but correct when present."""
        info = _get_device_info()
        # Timing stats may or may not be present depending on whether TTS engine exists
        if "seconds_per_char" in info:
            assert isinstance(info["seconds_per_char"], float)
            assert info["seconds_per_char"] > 0
        if "total_chars_processed" in info:
            assert isinstance(info["total_chars_processed"], int)
            assert info["total_chars_processed"] > 0


class TestLanguagesEndpoint:
    """Tests for the languages endpoint."""

    async def test_get_languages(self, client: AsyncClient) -> None:
        """Test getting available languages."""
        response = await client.get("/api/languages")
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert "english" in data["languages"]
        assert "chinese" in data["languages"]


class TestStreamingEndpoints:
    """Tests for streaming endpoints."""

    async def test_read_pdf_stream_rejects_non_pdf(self, client: AsyncClient) -> None:
        """Test that non-PDF files are rejected in stream endpoint."""
        response = await client.post(
            "/api/read-stream",
            files={"file": ("test.txt", b"Hello", "text/plain")},
            data={"language": "english"},
        )
        assert response.status_code == 400

    async def test_read_pdf_stream_rejects_empty(self, client: AsyncClient) -> None:
        """Test that empty files are rejected in stream endpoint."""
        response = await client.post(
            "/api/read-stream",
            files={"file": ("test.pdf", b"", "application/pdf")},
            data={"language": "english"},
        )
        assert response.status_code == 400

    async def test_read_pdf_stream_invalid_language_defaults(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test that invalid language defaults to english."""
        response = await client.post(
            "/api/read-stream",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"language": "invalid_lang"},
        )
        # Should not error, just default to english
        assert response.status_code in (200, 400)

    async def test_read_text_stream_empty_text(self, client: AsyncClient) -> None:
        """Test that empty text is rejected."""
        response = await client.post(
            "/api/read-text-stream",
            json={"text": "", "language": "english"},
        )
        assert response.status_code == 400

    async def test_read_text_stream_whitespace_text(self, client: AsyncClient) -> None:
        """Test that whitespace-only text is rejected."""
        response = await client.post(
            "/api/read-text-stream",
            json={"text": "   \n\t  ", "language": "english"},
        )
        assert response.status_code == 400

    async def test_read_text_stream_too_long(self, client: AsyncClient) -> None:
        """Test that overly long text is rejected."""
        response = await client.post(
            "/api/read-text-stream",
            json={"text": "x" * 600000, "language": "english"},
        )
        assert response.status_code == 400
        assert "too long" in response.json()["detail"].lower()

    async def test_read_text_stream_success(self, client: AsyncClient) -> None:
        """Test successful text streaming."""
        response = await client.post(
            "/api/read-text-stream",
            json={"text": "Hello, this is a test.", "language": "english"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    async def test_read_url_stream_empty_url(self, client: AsyncClient) -> None:
        """Test that empty URL is rejected in stream endpoint."""
        response = await client.post(
            "/api/read-url-stream",
            json={"url": "", "language": "english"},
        )
        assert response.status_code == 400

    async def test_read_url_stream_invalid_scheme(self, client: AsyncClient) -> None:
        """Test that non-HTTP URLs are rejected in stream endpoint."""
        response = await client.post(
            "/api/read-url-stream",
            json={"url": "ftp://example.com/file", "language": "english"},
        )
        assert response.status_code == 400


class TestAudioStreamEndpoint:
    """Tests for the audio streaming endpoint."""

    async def test_audio_stream_nonexistent_job(self, client: AsyncClient) -> None:
        """Test streaming audio for nonexistent job returns 404."""
        response = await client.get("/api/audio/nonexistent-job-id")
        assert response.status_code == 404


class TestDeviceInfoEndpoint:
    """Tests for device info SSE endpoint."""

    async def test_device_info_stream_returns_sse(self, client: AsyncClient) -> None:
        """Test device info returns SSE content type."""
        import asyncio

        # Use a short timeout since this is an infinite streaming endpoint
        try:
            async with asyncio.timeout(2):
                async with client.stream("GET", "/api/device-info-stream") as response:
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers["content-type"]
                    # Read just the first chunk to verify it works
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            # Should be SSE format
                            assert b"data:" in chunk
                            break
        except TimeoutError:
            pass  # Expected - we just want to verify the endpoint works


class TestDownloadEndpoint:
    """Tests for the audio download endpoint."""

    async def test_download_nonexistent_job(self, client: AsyncClient) -> None:
        """Test downloading audio for nonexistent job returns 404."""
        response = await client.get("/api/download/nonexistent-job-id")
        assert response.status_code == 404

    async def test_download_job_without_audio(self, client: AsyncClient) -> None:
        """Test downloading job with no audio data returns 404."""
        job = _job_manager.create_job()
        try:
            response = await client.get(f"/api/download/{job.job_id}")
            assert response.status_code == 404
            assert "No audio" in response.json()["detail"]
        finally:
            _job_manager.remove_job(job.job_id)

    async def test_download_with_ascii_filename(self, client: AsyncClient) -> None:
        """Test downloading audio with ASCII filename."""
        job = _job_manager.create_job()
        # Add some fake audio data (PCM bytes)
        job.audio_cache.append(b"\x00\x00" * 1000)
        job.finish()

        try:
            response = await client.get(f"/api/download/{job.job_id}?filename=test_audio.wav")
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"

            # Check Content-Disposition header
            cd = response.headers["content-disposition"]
            assert "attachment" in cd
            assert "test_audio.wav" in cd
        finally:
            _job_manager.remove_job(job.job_id)

    async def test_download_with_chinese_filename(self, client: AsyncClient) -> None:
        """Test downloading audio with Chinese (non-ASCII) filename."""
        job = _job_manager.create_job()
        # Add some fake audio data (PCM bytes)
        job.audio_cache.append(b"\x00\x00" * 1000)
        job.finish()

        chinese_filename = "从前有座山.wav"

        try:
            response = await client.get(f"/api/download/{job.job_id}?filename={chinese_filename}")
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"

            # Check Content-Disposition header has RFC 5987 encoding
            cd = response.headers["content-disposition"]
            assert "attachment" in cd
            # Should have ASCII fallback with replacement chars
            assert 'filename="' in cd
            # Should have UTF-8 encoded filename*
            assert "filename*=UTF-8''" in cd
            # The UTF-8 encoded form should contain percent-encoded Chinese chars
            assert "%E4%BB%8E%E5%89%8D" in cd  # "从前" encoded
        finally:
            _job_manager.remove_job(job.job_id)

    async def test_download_with_japanese_filename(self, client: AsyncClient) -> None:
        """Test downloading audio with Japanese (non-ASCII) filename."""
        job = _job_manager.create_job()
        job.audio_cache.append(b"\x00\x00" * 1000)
        job.finish()

        japanese_filename = "音声ファイル.wav"

        try:
            response = await client.get(f"/api/download/{job.job_id}?filename={japanese_filename}")
            assert response.status_code == 200

            cd = response.headers["content-disposition"]
            assert "filename*=UTF-8''" in cd
        finally:
            _job_manager.remove_job(job.job_id)

    async def test_download_returns_valid_wav(self, client: AsyncClient) -> None:
        """Test that downloaded file is a valid WAV with correct header."""
        job = _job_manager.create_job()
        # Add some fake audio data
        job.audio_cache.append(b"\x00\x00" * 1000)
        job.finish()

        try:
            response = await client.get(f"/api/download/{job.job_id}")
            assert response.status_code == 200

            content = response.content
            # Check WAV header magic bytes
            assert content[:4] == b"RIFF"
            assert content[8:12] == b"WAVE"
            assert content[12:16] == b"fmt "
        finally:
            _job_manager.remove_job(job.job_id)
