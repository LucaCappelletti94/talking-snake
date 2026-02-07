"""PDF text extraction and cleaning for TTS processing."""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTPage, LTTextBoxHorizontal, LTTextLineHorizontal


@dataclass
class TextBlock:
    """A block of text with positional metadata."""

    text: str
    y_ratio: float  # 0.0 = bottom, 1.0 = top
    font_size: float
    page_num: int


def extract_text_blocks(pdf_bytes: bytes) -> list[TextBlock]:
    """Extract text blocks from PDF with positional information.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        List of TextBlock objects with text and metadata.
    """
    blocks: list[TextBlock] = []
    pdf_file = io.BytesIO(pdf_bytes)

    laparams = LAParams(
        line_margin=0.5,
        word_margin=0.1,
        char_margin=2.0,
        boxes_flow=0.5,
    )

    for page_num, page_layout in enumerate(extract_pages(pdf_file, laparams=laparams), start=1):
        if not isinstance(page_layout, LTPage):
            continue

        page_height = page_layout.height

        for element in page_layout:
            if not isinstance(element, LTTextBoxHorizontal):
                continue

            text = element.get_text().strip()
            if not text:
                continue

            # Calculate Y position as ratio (0=bottom, 1=top)
            y_ratio = element.y0 / page_height if page_height > 0 else 0.5

            # Extract average font size from characters
            font_sizes: list[float] = []
            for line in element:
                if isinstance(line, LTTextLineHorizontal):
                    for char in line:
                        if isinstance(char, LTChar):
                            font_sizes.append(char.size)

            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10.0

            blocks.append(
                TextBlock(
                    text=text,
                    y_ratio=y_ratio,
                    font_size=avg_font_size,
                    page_num=page_num,
                )
            )

    return blocks


def get_page_count(pdf_bytes: bytes) -> int:
    """Get the number of pages in a PDF.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Number of pages in the PDF.
    """
    pdf_file = io.BytesIO(pdf_bytes)
    laparams = LAParams()
    page_count = sum(1 for _ in extract_pages(pdf_file, laparams=laparams))
    return page_count


def extract_text(pdf_bytes: bytes) -> str:
    """Extract and clean text from a PDF file.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Cleaned text suitable for TTS.
    """
    blocks = extract_text_blocks(pdf_bytes)
    if not blocks:
        return ""

    cleaned_blocks = clean_text_blocks(blocks)
    text = "\n\n".join(block.text for block in cleaned_blocks)

    # Apply TTS-specific normalization
    return normalize_for_tts(text)


def clean_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Remove headers, footers, page numbers, and other artifacts.

    Applies multiple heuristics:
    1. Remove blocks in top/bottom margins (likely headers/footers)
    2. Remove repeated text across pages (likely running headers)
    3. Remove standalone page numbers
    4. Remove very short lines that look like artifacts

    Args:
        blocks: List of TextBlock objects.

    Returns:
        Filtered list of TextBlock objects.
    """
    if not blocks:
        return []

    # Find repeated text patterns (headers/footers)
    text_counts = Counter(block.text for block in blocks)
    total_pages = max(block.page_num for block in blocks)
    repeated_threshold = max(2, total_pages // 2)
    repeated_texts = {text for text, count in text_counts.items() if count >= repeated_threshold}

    # Calculate median font size for filtering
    font_sizes = sorted(block.font_size for block in blocks)
    median_font_size = font_sizes[len(font_sizes) // 2] if font_sizes else 10.0

    cleaned: list[TextBlock] = []

    for block in blocks:
        # Skip if in header zone (top 10%)
        if block.y_ratio > 0.90:
            continue

        # Skip if in footer zone (bottom 10%)
        if block.y_ratio < 0.10:
            continue

        # Skip repeated text (running headers/footers)
        if block.text in repeated_texts:
            continue

        # Skip standalone page numbers
        if is_page_number(block.text):
            continue

        # Skip very short lines with small font (likely captions/footnotes)
        if len(block.text) < 20 and block.font_size < median_font_size * 0.8:
            continue

        cleaned.append(block)

    return cleaned


def is_page_number(text: str) -> bool:
    """Check if text is likely a page number.

    Args:
        text: Text to check.

    Returns:
        True if text appears to be a page number.
    """
    text = text.strip()

    # Pure number
    if text.isdigit():
        return True

    # Roman numerals
    if re.match(r"^[ivxlcdmIVXLCDM]+$", text):
        return True

    # "Page N" or "N of M" patterns
    if re.match(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", text, re.IGNORECASE):
        return True

    # "- N -" pattern
    if re.match(r"^[-–—]\s*\d+\s*[-–—]$", text):
        return True

    return False


def clean_text(text: str) -> str:
    """Clean raw text for TTS processing.

    This is a simpler function for cleaning already-extracted text,
    without the positional information.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text suitable for TTS.
    """
    lines = text.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip standalone page numbers
        if is_page_number(line):
            continue

        # Skip very short lines (likely artifacts)
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    # Rejoin with proper spacing
    result = "\n".join(cleaned_lines)

    # === FIX HYPHENATED/SPLIT WORDS ===
    # These are words broken across lines, common in PDFs and web content

    # Pattern 1: word-\nword (hyphen at end of line) -> rejoin word
    result = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", result)

    # Pattern 2: word-\n  word (hyphen + newline + spaces)
    result = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", result)

    # Pattern 3: word- word (hyphen + space, often from copy-paste)
    result = re.sub(r"(\w)- (\w)", r"\1\2", result)

    # Pattern 4: Lines ending with hyphen followed by lowercase (likely continuation)
    result = re.sub(r"-\n([a-z])", r"\1", result)

    # === FIX LINE BREAK ARTIFACTS ===
    # Join lines that don't end with sentence-ending punctuation
    # This handles text that was wrapped at fixed width

    # Replace single newlines (not paragraph breaks) with spaces
    # Keep double newlines as paragraph separators
    result = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", result)

    # Normalize whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]+", " ", result)

    # Apply TTS-specific normalization
    result = normalize_for_tts(result)

    return result.strip()


def normalize_for_tts(text: str) -> str:
    """Normalize text for natural TTS pronunciation.

    Handles special characters, punctuation, and formatting that can
    cause TTS models to slow down or mispronounce.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text optimized for TTS.
    """
    # === CODE AND TECHNICAL CONTENT ===
    # Handle common programming patterns that read poorly

    # === REMOVE URLS AND TECHNICAL STRINGS FIRST ===
    # URLs (various formats) - remove completely
    text = re.sub(r"https?://[^\s<>\"')\]]+", "", text)
    text = re.sub(r"www\.[^\s<>\"')\]]+", "", text)
    text = re.sub(r"ftp://[^\s<>\"')\]]+", "", text)

    # UUIDs (with or without dashes) - must come before git hash pattern
    uuid_pattern = (
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    )
    text = re.sub(uuid_pattern, "", text)

    # Git commit hashes (7-40 hex chars standalone)
    text = re.sub(r"(?<![a-zA-Z0-9])[0-9a-f]{7,40}(?![a-zA-Z0-9])", "", text, flags=re.IGNORECASE)

    # Hex color codes (#fff, #ffffff)
    text = re.sub(r"#[0-9a-fA-F]{3,8}\b", "", text)

    # Long hex/base64 strings (likely encoded data)
    text = re.sub(r"\b[A-Za-z0-9+/]{20,}={0,2}\b", "", text)

    # File paths (Unix and Windows style)
    text = re.sub(r"[/\\][\w./\\-]+\.\w+", "", text)

    # IP addresses
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "", text)

    # Port numbers after colon
    text = re.sub(r":\d{2,5}\b", "", text)

    # Remove email addresses
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text)

    # SHA/MD5 style hashes with prefix
    text = re.sub(r"\b(sha\d*|md5|hash)[:\s]*[0-9a-f]+\b", "", text, flags=re.IGNORECASE)

    # CamelCase: split into words (e.g., "getUserName" -> "get User Name")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # snake_case: replace underscores with spaces
    text = re.sub(r"(\w)_(\w)", r"\1 \2", text)

    # Function calls: "func()" -> "func"
    text = re.sub(r"(\w+)\(\)", r"\1", text)

    # Arrow functions/operators: -> and =>
    text = text.replace("->", " returns ")
    text = text.replace("=>", " arrow ")

    # Common code operators spoken naturally
    text = text.replace("!=", " not equals ")
    text = text.replace("==", " equals ")
    text = text.replace("===", " strictly equals ")
    text = text.replace("!==", " strictly not equals ")
    text = text.replace("&&", " and ")
    text = text.replace("||", " or ")
    text = text.replace("++", " increment ")
    text = text.replace("--", " decrement ")

    # File extensions: ".py" -> " dot py" (only for common extensions)
    ext_pattern = r"\.(py|js|ts|html|css|json|xml|md|txt|csv|pdf)\b"
    text = re.sub(ext_pattern, r" dot \1", text, flags=re.IGNORECASE)

    # Remove standalone hashes/pound signs (not hashtags)
    text = re.sub(r"(?<!\w)#(?!\w)", "", text)

    # Backticks (often used in markdown for code)
    text = text.replace("`", "")

    # Triple quotes
    text = text.replace('"""', "")
    text = text.replace("'''", "")

    # === UNICODE NORMALIZATION ===
    # Convert smart quotes to simple quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace("„", '"').replace("‟", '"')

    # Normalize dashes to standard hyphen or remove
    text = text.replace("–", "-")  # en-dash
    text = text.replace("—", " - ")  # em-dash (add spaces for pause)
    text = text.replace("―", " - ")  # horizontal bar
    text = text.replace("‐", "-")  # Unicode hyphen
    text = text.replace("‑", "-")  # non-breaking hyphen
    text = text.replace("⁃", "-")  # hyphen bullet
    text = text.replace("−", "-")  # minus sign

    # Normalize ellipsis
    text = text.replace("…", "...")
    text = re.sub(r"\.{4,}", "...", text)  # Limit to 3 dots

    # Normalize other Unicode punctuation
    text = text.replace("•", ",")  # Bullet points
    text = text.replace("·", " ")  # Middle dot
    text = text.replace("‧", " ")  # Hyphenation point
    text = text.replace("※", " ")  # Reference mark
    text = text.replace("†", "")  # Dagger (footnote)
    text = text.replace("‡", "")  # Double dagger
    text = text.replace("§", "section ")
    text = text.replace("¶", "")  # Pilcrow
    text = text.replace("©", "copyright ")
    text = text.replace("®", " registered ")
    text = text.replace("™", " trademark ")
    text = text.replace("°", " degrees ")

    # === SPACING AROUND PUNCTUATION ===
    # Ensure proper spacing around dashes used as separators
    text = re.sub(r"\s*-\s*-\s*", " - ", text)  # Double dash
    text = re.sub(r"(\w)\s*-\s*(\w)", r"\1 - \2", text)  # Word-dash-word with spaces

    # Fix missing space after punctuation
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    text = re.sub(r",([A-Za-z])", r", \1", text)

    # Fix multiple punctuation marks
    text = re.sub(r"[,]{2,}", ",", text)
    text = re.sub(r"[;]{2,}", ";", text)
    text = re.sub(r"[:]{2,}", ":", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)

    # === NUMBERS AND SPECIAL NOTATIONS ===
    # Convert common fractions
    text = text.replace("½", " one half ")
    text = text.replace("⅓", " one third ")
    text = text.replace("⅔", " two thirds ")
    text = text.replace("¼", " one quarter ")
    text = text.replace("¾", " three quarters ")
    text = text.replace("⅕", " one fifth ")
    text = text.replace("⅖", " two fifths ")
    text = text.replace("⅗", " three fifths ")
    text = text.replace("⅘", " four fifths ")
    text = text.replace("⅙", " one sixth ")
    text = text.replace("⅚", " five sixths ")
    text = text.replace("⅛", " one eighth ")
    text = text.replace("⅜", " three eighths ")
    text = text.replace("⅝", " five eighths ")
    text = text.replace("⅞", " seven eighths ")

    # Handle percentage and math symbols
    text = text.replace("%", " percent")
    text = text.replace("&", " and ")
    text = text.replace("+", " plus ")
    text = text.replace("=", " equals ")
    text = text.replace("<", " less than ")
    text = text.replace(">", " greater than ")
    text = text.replace("≤", " less than or equal to ")
    text = text.replace("≥", " greater than or equal to ")
    text = text.replace("≠", " not equal to ")
    text = text.replace("±", " plus or minus ")
    text = text.replace("×", " times ")
    text = text.replace("÷", " divided by ")

    # === ABBREVIATIONS AND SPECIAL CASES ===
    # Common abbreviations that might cause issues
    text = re.sub(r"\be\.g\.", "for example", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.e\.", "that is", text, flags=re.IGNORECASE)
    text = re.sub(r"\betc\.", "etcetera", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvs\.", "versus", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDr\.", "Doctor", text)
    text = re.sub(r"\bMr\.", "Mister", text)
    text = re.sub(r"\bMrs\.", "Missus", text)
    text = re.sub(r"\bMs\.", "Miss", text)
    text = re.sub(r"\bProf\.", "Professor", text)
    text = re.sub(r"\bSt\.", "Saint", text)
    text = re.sub(r"\bNo\.\s*(\d)", r"Number \1", text)
    text = re.sub(r"\bFig\.", "Figure", text, flags=re.IGNORECASE)
    text = re.sub(r"\bVol\.", "Volume", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpp\.", "pages", text, flags=re.IGNORECASE)
    text = re.sub(r"\bp\.\s*(\d)", r"page \1", text, flags=re.IGNORECASE)

    # === BRACKETS AND PARENTHESES ===
    # Remove or simplify brackets that might cause pauses
    text = re.sub(r"\[([^\]]+)\]", r"(\1)", text)  # Square to round
    text = re.sub(r"\{([^}]+)\}", r"(\1)", text)  # Curly to round

    # Remove citation numbers like [1], [2,3], [1-5]
    text = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", text)
    text = re.sub(r"\(\d+(?:[-,]\d+)*\)", "", text)

    # === CLEANUP ===
    # Remove standalone special characters
    text = re.sub(r"\s+[#@*^~`|\\]+\s+", " ", text)

    # Remove content in angle brackets (often HTML/XML artifacts)
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Ensure space after punctuation (but not before another punctuation)
    text = re.sub(r"([.,;:!?])([^\s.,;:!?'\"])", r"\1 \2", text)

    # Remove leading/trailing whitespace from lines
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove empty lines that resulted from cleaning
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text
