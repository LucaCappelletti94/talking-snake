"""Test configuration and shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from talking_snake.tts import MockTTSEngine

if TYPE_CHECKING:
    pass


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom pytest options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g., real TTS model inference)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow tests unless --run-slow is provided."""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def mock_tts_engine() -> MockTTSEngine:
    """Provide a mock TTS engine for testing."""
    return MockTTSEngine(sample_rate=24000)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a minimal valid PDF for testing.

    This creates a simple single-page PDF with some text content.
    """
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 178 >>
stream
BT
/F1 12 Tf
50 700 Td
(This is a test document for the Reader application.) Tj
0 -20 Td
(It contains multiple lines of text that should be extracted.) Tj
0 -20 Td
(The text extraction module should handle this correctly.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000496 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
573
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_pdf_with_headers_bytes() -> bytes:
    """Generate a PDF with headers and footers for testing cleanup."""
    # This simulates a multi-page document with repeated headers/footers
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 280 >>
stream
BT
/F1 10 Tf
50 760 Td
(Document Title - Header) Tj
/F1 12 Tf
50 600 Td
(This is the main content of page one.) Tj
0 -20 Td
(It contains important information that should be read aloud.) Tj
/F1 10 Tf
50 50 Td
(Page 1) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
6 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 7 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
7 0 obj
<< /Length 280 >>
stream
BT
/F1 10 Tf
50 760 Td
(Document Title - Header) Tj
/F1 12 Tf
50 600 Td
(This is the main content of page two.) Tj
0 -20 Td
(More valuable content that needs text to speech conversion.) Tj
/F1 10 Tf
50 50 Td
(Page 2) Tj
ET
endstream
endobj
xref
0 8
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000119 00000 n
0000000270 00000 n
0000000602 00000 n
0000000671 00000 n
0000000822 00000 n
trailer
<< /Size 8 /Root 1 0 R >>
startxref
1154
%%EOF"""
    return pdf_content


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Generate a PDF with no extractable text."""
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << >> >>
endobj
4 0 obj
<< /Length 0 >>
stream
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
264
%%EOF"""
    return pdf_content
