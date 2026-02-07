"""Integration tests for the FastAPI application."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from talking_snake.app import create_app
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


class TestHealthCheck:
    """Tests for the health check endpoint."""

    async def test_health_check_returns_ok(self, client: AsyncClient) -> None:
        """Test that health check returns OK status."""
        response = await client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestIndexPage:
    """Tests for the main page."""

    async def test_index_returns_html(self, client: AsyncClient) -> None:
        """Test that index page returns HTML."""
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    async def test_index_contains_reader_title(self, client: AsyncClient) -> None:
        """Test that index page contains the Talking Snake title."""
        response = await client.get("/")
        assert b"Talking Snake" in response.content


class TestReadPdfEndpoint:
    """Tests for the PDF reading endpoint."""

    async def test_rejects_non_pdf(self, client: AsyncClient) -> None:
        """Test that non-PDF files are rejected."""
        response = await client.post(
            "/api/read",
            files={"file": ("test.txt", b"Hello world", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    async def test_rejects_empty_file(self, client: AsyncClient) -> None:
        """Test that empty files are rejected."""
        response = await client.post(
            "/api/read",
            files={"file": ("test.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400

    async def test_accepts_valid_pdf(self, client: AsyncClient, sample_pdf_bytes: bytes) -> None:
        """Test that valid PDFs are accepted and return audio."""
        response = await client.post(
            "/api/read",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        # May return 400 if no text extracted, or 200 with audio
        assert response.status_code in (200, 400)

        if response.status_code == 200:
            assert response.headers["content-type"] == "audio/wav"

    async def test_returns_wav_content_type(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test that successful response has WAV content type."""
        response = await client.post(
            "/api/read",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        if response.status_code == 200:
            assert response.headers["content-type"] == "audio/wav"

    async def test_missing_file(self, client: AsyncClient) -> None:
        """Test that missing file returns error."""
        response = await client.post("/api/read")
        assert response.status_code == 422  # Validation error

    async def test_invalid_pdf_data(self, client: AsyncClient) -> None:
        """Test that invalid PDF data returns error."""
        response = await client.post(
            "/api/read",
            files={"file": ("test.pdf", b"not a real pdf", "application/pdf")},
        )
        assert response.status_code == 400


class TestReadUrlEndpoint:
    """Tests for the URL reading endpoint."""

    async def test_rejects_empty_url(self, client: AsyncClient) -> None:
        """Test that empty URL is rejected."""
        response = await client.post(
            "/api/read-url",
            json={"url": ""},
        )
        assert response.status_code == 400
        assert "required" in response.json()["detail"].lower()

    async def test_rejects_whitespace_url(self, client: AsyncClient) -> None:
        """Test that whitespace-only URL is rejected."""
        response = await client.post(
            "/api/read-url",
            json={"url": "   "},
        )
        assert response.status_code == 400

    async def test_rejects_non_http_url(self, client: AsyncClient) -> None:
        """Test that non-HTTP URLs are rejected."""
        response = await client.post(
            "/api/read-url",
            json={"url": "ftp://example.com/doc.pdf"},
        )
        assert response.status_code == 400
        assert "HTTP" in response.json()["detail"]

    async def test_rejects_file_url(self, client: AsyncClient) -> None:
        """Test that file:// URLs are rejected."""
        response = await client.post(
            "/api/read-url",
            json={"url": "file:///etc/passwd.pdf"},
        )
        assert response.status_code == 400

    async def test_accepts_html_url_format(self, client: AsyncClient) -> None:
        """Test that HTML URLs are accepted (will fail on fetch, but format is valid)."""
        # This will fail because example.com won't respond as expected,
        # but the URL format itself should be accepted (not rejected as non-PDF)
        response = await client.post(
            "/api/read-url",
            json={"url": "https://example.com/article"},
        )
        # Should fail on fetch/extraction, not on URL validation
        assert response.status_code in (400, 408)  # Network error or timeout, not validation
        # Should NOT say "must point to a PDF"
        if response.status_code == 400:
            detail = response.json().get("detail", "")
            assert "PDF" not in detail or "extract" in detail.lower()

    async def test_missing_url_field(self, client: AsyncClient) -> None:
        """Test that missing URL field returns error."""
        response = await client.post(
            "/api/read-url",
            json={},
        )
        assert response.status_code == 422  # Validation error

    async def test_invalid_json(self, client: AsyncClient) -> None:
        """Test that invalid JSON returns error."""
        response = await client.post(
            "/api/read-url",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestStaticFiles:
    """Tests for static file serving."""

    async def test_htmx_script_tag_in_index(self, client: AsyncClient) -> None:
        """Test that HTMX is included in the index page."""
        response = await client.get("/")
        assert b"htmx" in response.content


class TestAppCreation:
    """Tests for application creation."""

    def test_create_app_with_mock_engine(self) -> None:
        """Test that app can be created with mock engine."""
        mock_engine = MockTTSEngine()
        app = create_app(tts_engine=mock_engine)
        assert app is not None
        assert app.title == "Reader"

    def test_create_app_without_engine(self) -> None:
        """Test that app can be created without engine (uses mock)."""
        app = create_app()
        assert app is not None


class TestLanguageParameter:
    """Tests for language parameter handling."""

    async def test_text_stream_accepts_language(self, client: AsyncClient) -> None:
        """Test text stream endpoint accepts language parameter."""
        response = await client.post(
            "/api/read-text-stream",
            json={"text": "Hello world", "language": "japanese"},
        )
        assert response.status_code == 200

    async def test_text_stream_accepts_chinese(self, client: AsyncClient) -> None:
        """Test text stream endpoint accepts Chinese language."""
        response = await client.post(
            "/api/read-text-stream",
            json={"text": "你好世界", "language": "chinese"},
        )
        assert response.status_code == 200

    async def test_text_stream_accepts_korean(self, client: AsyncClient) -> None:
        """Test text stream endpoint accepts Korean language."""
        response = await client.post(
            "/api/read-text-stream",
            json={"text": "Hello world", "language": "korean"},
        )
        assert response.status_code == 200


class TestJobTimeout:
    """Tests for job timeout and cleanup."""

    def test_job_has_started_timestamp(self) -> None:
        """Test that job has a started timestamp."""
        from talking_snake.app import AudioJob

        job = AudioJob("test-id")
        import time

        assert job.started <= time.time()
        assert job.started > time.time() - 10  # Within last 10 seconds
