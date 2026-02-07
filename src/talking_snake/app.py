"""FastAPI application for PDF-to-Speech server."""

from __future__ import annotations

import io
import json
import queue
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlparse

import httpx
import trafilatura
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from talking_snake.extract import clean_text, extract_text, get_page_count
from talking_snake.tts import (
    DEFAULT_CHUNK_SIZE,
    LANGUAGE_VOICES,
    TTS_STYLES,
    MockTTSEngine,
    TTSEngineProtocol,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# Request timeout for fetching URLs (seconds)
URL_FETCH_TIMEOUT = 60.0
# Maximum file size to fetch (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Initial estimate for time calculation before calibration
# This value is refined after the first chunk is processed
# RTX 4090 + flash-attn: ~0.001s/char, RTX 4090: ~0.002s/char, RTX 3060: ~0.005s/char
INITIAL_SECONDS_PER_CHAR = 0.002  # Optimistic GPU estimate, calibrates after first chunk

# Job timeout (seconds) - jobs are cleaned up after this time
JOB_TIMEOUT = 3600  # 1 hour


class AudioJob:
    """Represents an audio generation job with a queue for streaming."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self.audio_cache: list[bytes] = []  # Cache PCM chunks for replay/download
        self.started = time.time()
        self.completed = False
        self.stream_started = False  # Track if live stream has started
        self.error: str | None = None
        self.sample_rate = 24000  # Default, will be set by TTS engine
        self.header_sent = False
        self._total_pcm_bytes = 0  # Track total audio bytes for duration calc

    @property
    def audio_duration(self) -> float:
        """Calculate audio duration in seconds from cached PCM data."""
        # 16-bit mono audio: duration = bytes / (sample_rate * 2)
        return self._total_pcm_bytes / (self.sample_rate * 2)

    def put_audio(self, audio_bytes: bytes) -> None:
        """Add audio data to the queue and cache."""
        self.audio_queue.put(audio_bytes)
        # Cache the PCM data (strip WAV header if present)
        if audio_bytes[:4] == b"RIFF":
            pcm_data = audio_bytes[44:]
        else:
            pcm_data = audio_bytes
        self.audio_cache.append(pcm_data)
        self._total_pcm_bytes += len(pcm_data)

    def finish(self) -> None:
        """Signal that audio generation is complete."""
        self.completed = True
        self.audio_queue.put(None)  # Sentinel to signal end

    def set_error(self, error: str) -> None:
        """Set an error and finish the job."""
        self.error = error
        self.completed = True
        self.audio_queue.put(None)


class JobManager:
    """Manages audio generation jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, AudioJob] = {}
        self._lock = threading.Lock()

    def create_job(self) -> AudioJob:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        job = AudioJob(job_id)
        with self._lock:
            self._jobs[job_id] = job
            self._cleanup_old_jobs()
        return job

    def get_job(self, job_id: str) -> AudioJob | None:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove a job."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def _cleanup_old_jobs(self) -> None:
        """Remove jobs older than JOB_TIMEOUT."""
        now = time.time()
        to_remove = [jid for jid, job in self._jobs.items() if now - job.started > JOB_TIMEOUT]
        for jid in to_remove:
            del self._jobs[jid]


# Global job manager
_job_manager = JobManager()


class UrlRequest(BaseModel):
    """Request body for URL-based reading."""

    url: str
    language: str = "english"
    style: str = "technical"


class TextRequest(BaseModel):
    """Request body for direct text reading."""

    text: str
    language: str = "english"
    style: str = "technical"


class EstimateResponse(BaseModel):
    """Response for time estimation."""

    text_length: int
    chunk_count: int
    estimated_seconds: float
    estimated_minutes: float


# Global TTS engine instance (set during startup)
_tts_engine: TTSEngineProtocol | None = None


def create_app(tts_engine: TTSEngineProtocol | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        tts_engine: TTS engine to use. If None, uses MockTTSEngine.

    Returns:
        Configured FastAPI application.
    """
    global _tts_engine
    _tts_engine = tts_engine or MockTTSEngine()

    app = FastAPI(
        title="Reader",
        description="PDF-to-Speech web server - listen to any content",
        version="0.1.0",
    )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Register routes
    app.add_api_route("/", index, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/read", read_pdf, methods=["POST"])
    app.add_api_route("/api/read-url", read_url, methods=["POST"])
    app.add_api_route("/api/read-stream", read_pdf_stream, methods=["POST"])
    app.add_api_route("/api/read-url-stream", read_url_stream, methods=["POST"])
    app.add_api_route("/api/read-text-stream", read_text_stream, methods=["POST"])
    app.add_api_route("/api/audio/{job_id}", stream_audio, methods=["GET"])
    app.add_api_route("/api/download/{job_id}", download_audio, methods=["GET"])
    app.add_api_route("/api/languages", get_languages, methods=["GET"])
    app.add_api_route("/api/device-info-stream", stream_device_info, methods=["GET"])
    app.add_api_route("/api/health", health_check, methods=["GET"])

    return app


async def index(request: Request) -> HTMLResponse:
    """Serve the main page.

    Args:
        request: The incoming request.

    Returns:
        HTML response with the main page.
    """
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"

    if not index_file.exists():
        return HTMLResponse(
            content="<h1>Reader</h1><p>Static files not found.</p>",
            status_code=200,
        )

    return HTMLResponse(content=index_file.read_text())


async def read_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    """Read a PDF and return synthesized speech.

    Args:
        file: Uploaded PDF file.

    Returns:
        Streaming WAV audio response.

    Raises:
        HTTPException: If file is not a PDF or extraction fails.
    """
    if _tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Extract text
    try:
        text = extract_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF")

    # Stream TTS audio
    def generate_audio() -> Iterator[bytes]:
        assert _tts_engine is not None
        yield from _tts_engine.synthesize(text)

    return StreamingResponse(
        generate_audio(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{Path(file.filename).stem}.wav"',
        },
    )


async def read_url(request: UrlRequest) -> StreamingResponse:
    """Read content from a URL (PDF or web page) and return synthesized speech.

    For PDFs: extracts text and removes headers/footers/page numbers.
    For web pages: extracts main article content, removing navigation,
    sidebars, footers, ads, and other boilerplate.

    Args:
        request: Request containing the URL to fetch.

    Returns:
        Streaming WAV audio response.

    Raises:
        HTTPException: If URL is invalid, fetch fails, or extraction fails.
    """
    if _tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    # Validate URL
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only HTTP/HTTPS URLs are supported")

    # Determine if this is a PDF or web page
    is_pdf = parsed.path.lower().endswith(".pdf")

    # Fetch the content
    try:
        async with httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check content length if available
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024}MB",
                )

            content = response.content

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024}MB",
                )

            # Also check content-type header to detect PDFs served without .pdf extension
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type:
                is_pdf = True

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Request timed out while fetching URL")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch URL: HTTP {e.response.status_code}",
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if not content:
        raise HTTPException(status_code=400, detail="Empty content at URL")

    # Extract text based on content type
    if is_pdf:
        try:
            text = extract_text(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {e}")
    else:
        # Use trafilatura to extract main content from HTML
        # This removes navigation, sidebars, footers, ads, etc.
        try:
            extracted = trafilatura.extract(
                content,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True,
            )
            if extracted:
                # Apply additional cleaning for TTS
                text = clean_text(extracted)
            else:
                text = ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract page content: {e}")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No readable content found at URL")

    # Extract filename from URL for the response
    filename = Path(parsed.path).stem or parsed.netloc or "document"

    # Stream TTS audio
    def generate_audio() -> Iterator[bytes]:
        assert _tts_engine is not None
        yield from _tts_engine.synthesize(text)

    return StreamingResponse(
        generate_audio(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{filename}.wav"',
        },
    )


async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status information.
    """
    return {"status": "ok"}


async def get_languages() -> dict[str, list[str]]:
    """Get available languages.

    Returns:
        List of available language names.
    """
    return {"languages": list(LANGUAGE_VOICES.keys())}


def _get_device_info() -> dict:
    """Get device and model information with real-time memory stats.

    Returns:
        Device type, memory usage, and model info.
    """
    import shutil

    import psutil
    import torch

    info = {
        "device": "cpu",
        "device_name": "CPU",
        "memory_used_gb": 0,
        "memory_total_gb": 0,
        "memory_percent": 0,
        "batch_size": 1,
        "ram_used_gb": 0,
        "ram_total_gb": 0,
        "disk_free_gb": 0,
    }

    # Get RAM info
    ram = psutil.virtual_memory()
    info["ram_used_gb"] = round(ram.used / 1024**3, 1)
    info["ram_total_gb"] = round(ram.total / 1024**3, 1)

    # Get disk free space
    disk = shutil.disk_usage("/")
    info["disk_free_gb"] = round(disk.free / 1024**3, 1)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Use reserved memory for more accurate GPU usage (includes PyTorch cache)
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        total = props.total_memory

        # Show reserved memory (what's actually held by PyTorch)
        used = max(reserved, allocated)

        info["device"] = "cuda"
        info["device_name"] = props.name
        info["memory_used_gb"] = round(used / 1024**3, 1)
        info["memory_total_gb"] = round(total / 1024**3, 1)
        info["memory_percent"] = round((used / total) * 100, 1) if total > 0 else 0
        # Also include allocated for debugging
        info["memory_allocated_gb"] = round(allocated / 1024**3, 1)

    if _tts_engine is not None:
        info["batch_size"] = getattr(_tts_engine, "batch_size", 1)
        info["chunk_size"] = getattr(_tts_engine, "chunk_size", 800)
        # Include model state
        info["model_state"] = getattr(_tts_engine, "model_state", "unknown")
        # Include timing stats
        seconds_per_char = getattr(_tts_engine, "seconds_per_char", None)
        if seconds_per_char is not None:
            info["seconds_per_char"] = round(seconds_per_char, 4)
        total_chars = getattr(_tts_engine, "total_chars_processed", 0)
        if total_chars > 0:
            info["total_chars_processed"] = total_chars

    return info


async def stream_device_info() -> StreamingResponse:
    """Stream device info updates via SSE.

    Returns:
        SSE stream with device info updates every 3 seconds.
    """
    import asyncio
    from collections.abc import AsyncIterator
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)

    async def generate_events() -> AsyncIterator[str]:
        """Generate SSE events for device info."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Run torch calls in executor to avoid blocking
                info = await loop.run_in_executor(executor, _get_device_info)
                yield f"data: {json.dumps(info)}\n\n"
            except Exception as e:
                # Send error info but continue
                yield f'data: {{"error": "{e!s}"}}\n\n'
            await asyncio.sleep(3)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _estimate_time(text: str, seconds_per_char: float | None = None) -> tuple[int, float]:
    """Estimate processing time for text.

    Args:
        text: Text to process.
        seconds_per_char: Calibrated rate (defaults to initial estimate).

    Returns:
        Tuple of (chunk_count, estimated_seconds).
    """
    if seconds_per_char is None:
        seconds_per_char = INITIAL_SECONDS_PER_CHAR
    # Count chunks (500 chars per chunk approximately)
    chunk_count = max(1, len(text) // 500 + (1 if len(text) % 500 else 0))
    estimated_seconds = len(text) * seconds_per_char
    return chunk_count, estimated_seconds


def _create_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16) -> bytes:
    """Create a WAV header for streaming (unknown length).

    Uses maximum possible file size since we don't know the final length.

    Args:
        sample_rate: Audio sample rate.
        bits_per_sample: Bits per sample.

    Returns:
        WAV header bytes.
    """
    channels = 1
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    # Use maximum size for streaming (will be truncated on close)
    max_size = 0x7FFFFFFF

    header = io.BytesIO()
    header.write(b"RIFF")
    header.write(struct.pack("<I", max_size))
    header.write(b"WAVE")
    header.write(b"fmt ")
    header.write(struct.pack("<I", 16))  # fmt chunk size
    header.write(struct.pack("<H", 1))  # PCM format
    header.write(struct.pack("<H", channels))
    header.write(struct.pack("<I", sample_rate))
    header.write(struct.pack("<I", byte_rate))
    header.write(struct.pack("<H", block_align))
    header.write(struct.pack("<H", bits_per_sample))
    header.write(b"data")
    header.write(struct.pack("<I", max_size - 36))

    return header.getvalue()


def _generate_audio_to_job(
    job: AudioJob,
    text: str,
    tts_engine: TTSEngineProtocol,
    language: str = "english",
    style: str = "technical",
    doc_name: str = "document",
    doc_type: str = "text",
    page_count: int | None = None,
) -> Iterator[bytes]:
    """Generate audio with progress events via SSE, streaming audio to job queue.

    This function sends progress events via SSE while simultaneously writing
    audio data to the job's queue for streaming by another endpoint.
    Supports batched GPU inference for faster processing.

    Args:
        job: AudioJob to write audio data to.
        text: Text to synthesize.
        tts_engine: TTS engine to use.
        language: Language for TTS (english, chinese, japanese, korean).
        style: TTS style (technical, narrative, news, casual, academic).
        doc_name: Name of the document being processed.
        doc_type: Type of document (pdf, url, text).
        page_count: Number of pages (for PDFs).

    Yields:
        SSE events for progress.
    """
    import re

    # Apply language if the engine supports it
    if hasattr(tts_engine, "set_language"):
        tts_engine.set_language(language)

    # Apply style if the engine supports it
    if hasattr(tts_engine, "set_style"):
        tts_engine.set_style(style)

    # Get chunk size and batch size from engine
    chunk_size = getattr(tts_engine, "chunk_size", DEFAULT_CHUNK_SIZE)
    batch_size = getattr(tts_engine, "batch_size", 1)

    # Split text into chunks (same logic as TTS engine)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    total_chunks = len(chunks) if chunks else 1
    total_chars = sum(len(c) for c in chunks)

    # Use calibrated estimate if available, otherwise initial estimate
    seconds_per_char = getattr(tts_engine, "seconds_per_char", None) or INITIAL_SECONDS_PER_CHAR

    # Account for batch efficiency: processing N chunks in parallel is ~N times faster
    # The efficiency isn't perfectly linear, so use a conservative factor of sqrt(batch_size)
    batch_efficiency = batch_size**0.5 if batch_size > 1 else 1.0
    estimated_total = (total_chars * seconds_per_char) / batch_efficiency

    # Send initial progress event with job_id and batch info
    progress_data = {
        "type": "start",
        "job_id": job.job_id,
        "current": 0,
        "total": total_chunks,
        "percent": 0,
        "estimated_remaining": round(estimated_total, 1),
        "batch_size": batch_size,
        "doc_name": doc_name,
        "doc_type": doc_type,
        "page_count": page_count,
        "total_chars": total_chars,
        "status": f"Starting (batch size: {batch_size})...",
    }
    yield f"event: start\ndata: {json.dumps(progress_data)}\n\n".encode()

    # Generate audio - the TTS engine handles batching internally
    # We pass the full text and let it process in optimized batches
    start_time = time.time()
    chunks_processed = 0

    try:
        for audio_bytes in tts_engine.synthesize(text):
            # Write audio to job queue for streaming
            job.put_audio(audio_bytes)
            chunks_processed += 1

            # Calibrate time estimate
            elapsed = time.time() - start_time
            if chunks_processed > 0:
                time_per_chunk = elapsed / chunks_processed
                remaining_chunks = total_chunks - chunks_processed
                remaining = remaining_chunks * time_per_chunk
            else:
                remaining = estimated_total

            progress_data = {
                "type": "progress",
                "current": chunks_processed,
                "total": total_chunks,
                "percent": int((chunks_processed / total_chunks) * 100),
                "estimated_remaining": round(max(0, remaining), 1),
                "chars_processed": sum(
                    len(chunks[i]) for i in range(min(chunks_processed, len(chunks)))
                ),
                "total_chars": total_chars,
                "status": f"Processing chunk {chunks_processed}/{total_chunks}",
            }
            yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n".encode()

    except Exception as e:
        error_msg = f"TTS generation failed: {e!s}"
        error_data = {
            "type": "error",
            "message": error_msg,
            "chunk": chunks_processed + 1,
            "total_chunks": total_chunks,
        }
        job.set_error(error_msg)
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n".encode()
        return

    # Signal audio generation complete
    job.finish()

    # Send completion event with actual audio duration
    total_time = time.time() - start_time
    complete_data = {
        "type": "complete",
        "total_time": round(total_time, 1),
        "chunks_processed": chunks_processed,
        "batch_size": batch_size,
        "audio_duration": round(job.audio_duration, 2),
    }
    yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n".encode()


async def stream_audio(job_id: str) -> StreamingResponse:
    """Stream audio data for a job.

    This endpoint streams the raw WAV audio as it's being generated.
    The browser can start playing as soon as data arrives.
    First request streams live; subsequent requests return cached audio.

    Args:
        job_id: The job ID to stream audio for.

    Returns:
        Streaming WAV audio response.
    """
    job = _job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def generate_audio_live() -> Iterator[bytes]:
        """Stream audio live from queue (first request)."""
        job.stream_started = True
        # Send WAV header first
        yield _create_wav_header(sample_rate=24000)

        # Stream audio data as it becomes available
        while True:
            try:
                # Wait for audio data with timeout
                audio_data = job.audio_queue.get(timeout=300)  # 5 min timeout
                if audio_data is None:
                    # End of stream
                    break
                # Skip WAV headers from individual chunks, only send raw PCM
                if audio_data[:4] == b"RIFF":
                    yield audio_data[44:]
                else:
                    yield audio_data
            except queue.Empty:
                # Timeout waiting for data
                break

    def generate_audio_cached() -> Iterator[bytes]:
        """Stream audio from cache (subsequent requests)."""
        # Send WAV header first
        yield _create_wav_header(sample_rate=24000)
        # Send all cached chunks
        yield from job.audio_cache

    # Use live stream for first request, cached for subsequent
    if not job.stream_started:
        generator = generate_audio_live()
    else:
        generator = generate_audio_cached()

    return StreamingResponse(
        generator,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def download_audio(job_id: str, filename: str = "audio.wav") -> Response:
    """Download complete audio file for a job.

    This endpoint returns the full WAV file with correct headers for download.
    Only works after generation is complete.

    Args:
        job_id: The job ID to download audio for.
        filename: Suggested filename for download.

    Returns:
        Complete WAV audio file response.
    """
    job = _job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.audio_cache:
        raise HTTPException(status_code=404, detail="No audio available")

    # Combine all cached audio data
    audio_data = b"".join(job.audio_cache)

    # Create proper WAV header with actual size
    sample_rate = 24000
    bits_per_sample = 16
    channels = 1
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(audio_data)
    file_size = data_size + 36  # Header is 44 bytes, minus 8 for RIFF header

    header = io.BytesIO()
    header.write(b"RIFF")
    header.write(struct.pack("<I", file_size))
    header.write(b"WAVE")
    header.write(b"fmt ")
    header.write(struct.pack("<I", 16))  # fmt chunk size
    header.write(struct.pack("<H", 1))  # PCM format
    header.write(struct.pack("<H", channels))
    header.write(struct.pack("<I", sample_rate))
    header.write(struct.pack("<I", byte_rate))
    header.write(struct.pack("<H", block_align))
    header.write(struct.pack("<H", bits_per_sample))
    header.write(b"data")
    header.write(struct.pack("<I", data_size))

    wav_data = header.getvalue() + audio_data

    # RFC 5987 encoding for non-ASCII filenames
    # Use ASCII-safe fallback + UTF-8 encoded filename*
    safe_filename = filename.encode("ascii", "replace").decode("ascii")
    encoded_filename = quote(filename, safe="")

    return Response(
        content=wav_data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{safe_filename}"; ' f"filename*=UTF-8''{encoded_filename}"
            ),
            "Content-Length": str(len(wav_data)),
        },
    )


async def read_pdf_stream(
    file: UploadFile = File(...),
    language: str = Form("english"),
    style: str = Form("technical"),
) -> StreamingResponse:
    """Read a PDF with streaming progress updates.

    Returns SSE events for progress. Audio is streamed separately via /api/audio/{job_id}.

    Args:
        file: Uploaded PDF file.
        language: Language for TTS (english, chinese, japanese, korean).
        style: TTS style (technical, narrative, news, casual, academic).

    Returns:
        Streaming response with progress events including job_id.
    """
    if _tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    # Validate language
    if language not in LANGUAGE_VOICES:
        language = "english"

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        text = extract_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF")

    # Get page count for progress display
    try:
        page_count = get_page_count(pdf_bytes)
    except Exception:
        page_count = None

    # Create a job for this request
    job = _job_manager.create_job()

    return StreamingResponse(
        _generate_audio_to_job(
            job,
            text,
            _tts_engine,
            language,
            style,
            doc_name=file.filename or "document.pdf",
            doc_type="pdf",
            page_count=page_count,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def read_text_stream(request: TextRequest) -> StreamingResponse:
    """Read pasted text with streaming progress updates.

    Returns SSE events for progress. Audio is streamed separately via /api/audio/{job_id}.

    Args:
        request: Text request containing the text to read and language.

    Returns:
        Streaming response with progress events including job_id.
    """
    if _tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    text = request.text.strip()
    language = request.language if request.language in LANGUAGE_VOICES else "english"
    style = request.style if request.style in TTS_STYLES else "technical"

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if len(text) > 500000:  # ~500KB limit for pasted text
        raise HTTPException(status_code=400, detail="Text too long (max 500,000 characters)")

    # Apply text normalization
    text = clean_text(text)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text provided")

    # Generate doc name from first few words
    words = text.split()[:5]
    doc_name = " ".join(words)
    if len(doc_name) > 30:
        doc_name = doc_name[:30] + "..."
    elif len(words) == 5:
        doc_name = doc_name + "..."

    # Create a job for this request
    job = _job_manager.create_job()

    return StreamingResponse(
        _generate_audio_to_job(
            job,
            text,
            _tts_engine,
            language,
            style,
            doc_name=doc_name,
            doc_type="text",
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def read_url_stream(request: UrlRequest) -> StreamingResponse:
    """Read content from URL with streaming progress updates.

    Returns SSE events for progress. Audio is streamed separately via /api/audio/{job_id}.

    Args:
        request: URL request containing the URL to fetch and language.

    Returns:
        Streaming response with progress events including job_id.
    """
    if _tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    url = request.url.strip()
    language = request.language if request.language in LANGUAGE_VOICES else "english"
    style = request.style if request.style in TTS_STYLES else "technical"

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(status_code=400, detail="URL must use HTTP or HTTPS")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {e}")

    # Determine if this is a PDF or HTML page
    is_pdf = url.lower().endswith(".pdf")

    try:
        async with httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type:
                is_pdf = True

            if len(response.content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")

            content = response.content

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch URL: HTTP {e.response.status_code}"
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if is_pdf:
        try:
            text = extract_text(content)
            page_count = get_page_count(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {e}")
    else:
        page_count = None
        try:
            extracted = trafilatura.extract(
                content,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True,
            )
            if extracted:
                text = clean_text(extracted)
            else:
                text = ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract page content: {e}")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No readable content found at URL")

    # Extract document name from URL
    url_path = urlparse(url).path
    doc_name = url_path.split("/")[-1] if url_path else url
    if not doc_name or doc_name == "/":
        doc_name = urlparse(url).netloc

    # Create a job for this request
    job = _job_manager.create_job()

    return StreamingResponse(
        _generate_audio_to_job(
            job,
            text,
            _tts_engine,
            language,
            style,
            doc_name=doc_name,
            doc_type="pdf" if is_pdf else "url",
            page_count=page_count,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
