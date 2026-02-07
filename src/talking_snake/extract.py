"""PDF text extraction and cleaning for TTS processing."""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTPage,
    LTTextBoxHorizontal,
    LTTextLineHorizontal,
)


@dataclass
class TextBlock:
    """A block of text with positional metadata."""

    text: str
    y_ratio: float  # 0.0 = bottom, 1.0 = top
    font_size: float
    page_num: int
    x0: float = 0.0  # Left edge position for table detection
    x1: float = 0.0  # Right edge position for table detection


def _is_caption(text: str) -> bool:
    """Check if text is a figure/table caption.

    Captions typically start with:
    - "Figure 1:", "Fig. 2:", "Figure 1."
    - "Table 1:", "Table 2."
    - "Exhibit A:", "Chart 1:"
    - "Source:", "Note:", "Notes:"

    Args:
        text: Text to check.

    Returns:
        True if text appears to be a caption.
    """
    text = text.strip()
    if not text:
        return False

    # Common caption patterns (case-insensitive start)
    caption_patterns = [
        r"^fig(?:ure)?\.?\s*\d",
        r"^table\.?\s*\d",
        r"^exhibit\.?\s*[a-z0-9]",
        r"^chart\.?\s*\d",
        r"^graph\.?\s*\d",
        r"^diagram\.?\s*\d",
        r"^plate\.?\s*\d",
        r"^scheme\.?\s*\d",
        r"^box\.?\s*\d",
        r"^panel\.?\s*[a-z0-9]",
        r"^appendix\.?\s*[a-z0-9]",
        r"^source\s*:",
        r"^sources\s*:",
        r"^note\s*:",
        r"^notes\s*:",
        r"^data\s*:",
        r"^\*\s*p\s*[<>=]",  # Statistical notes like "* p < 0.05"
        r"^legend\s*:",
    ]

    text_lower = text.lower()
    for pattern in caption_patterns:
        if re.match(pattern, text_lower):
            return True

    return False


def _is_table_like_text(text: str) -> bool:
    """Check if text looks like table content.

    Tables often have:
    - Very short text fragments
    - Mostly numbers or single words
    - Lots of whitespace-separated values
    - Column headers or row labels
    - Short phrases without sentence structure

    Args:
        text: Text to check.

    Returns:
        True if the text appears to be table content.
    """
    text = text.strip()

    # Very short fragments are likely table cells
    if len(text) < 5:
        return True

    # Count numbers vs letters
    digits = sum(1 for c in text if c.isdigit())
    letters = sum(1 for c in text if c.isalpha())

    # Mostly numbers with few letters (like "123.45" or "2024")
    if digits > 0 and letters < 3 and digits >= letters:
        return True

    # Check for patterns common in tables
    # Multiple tab-separated or heavily spaced values
    if "\t" in text or "  " in text:
        parts = re.split(r"\s{2,}|\t", text)
        if len(parts) >= 3:
            # Multiple short parts suggests table row
            short_parts = sum(1 for p in parts if len(p.strip()) < 15)
            if short_parts >= len(parts) * 0.6:
                return True

    # Single words that look like column headers
    words = text.split()
    if len(words) == 1 and len(text) < 20:
        # Common table headers/labels
        table_keywords = {
            "total",
            "sum",
            "avg",
            "average",
            "mean",
            "count",
            "min",
            "max",
            "date",
            "time",
            "year",
            "month",
            "day",
            "name",
            "id",
            "no",
            "no.",
            "value",
            "amount",
            "price",
            "cost",
            "qty",
            "quantity",
            "unit",
            "row",
            "column",
            "col",
            "item",
            "description",
            "desc",
            "note",
            "status",
            "type",
            "category",
            "code",
            "ref",
            "reference",
        }
        if text.lower() in table_keywords:
            return True

    # Short phrases without sentence structure (likely table cells)
    # Table cells typically:
    # - Are short (< 50 chars)
    # - Don't end with sentence-ending punctuation
    # - Don't start with lowercase (unless very short)
    # - Have few words (< 8)
    if len(text) < 50 and len(words) < 8:
        # Doesn't end like a sentence
        if not text.rstrip().endswith((".", "!", "?", ":")):
            # Common table cell patterns
            text_lower = text.lower()

            # Technical/status phrases common in tables
            table_phrases = [
                "supported",
                "not supported",
                "yes",
                "no",
                "n/a",
                "none",
                "required",
                "optional",
                "enabled",
                "disabled",
                "active",
                "inactive",
                "read-only",
                "read only",
                "write",
                "read/write",
                "read-write",
                "must be",
                "can be",
                "should be",
                "will be",
                "available",
                "unavailable",
                "pending",
                "completed",
                "failed",
                "true",
                "false",
                "default",
                "custom",
                "manual",
                "automatic",
                "identical",
                "different",
                "same",
                "other",
            ]
            for phrase in table_phrases:
                if phrase in text_lower:
                    return True

            # Looks like a label or header (Title Case or ALL CAPS, short)
            if len(words) <= 4 and len(text) < 40:
                # Check if it's Title Case or contains common label patterns
                if text.istitle() or text.isupper():
                    return True
                # Two-three word phrases that look like labels
                if len(words) in (2, 3) and all(w[0].isupper() for w in words if w):
                    return True

    return False


def _filter_table_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Filter out blocks that appear to be part of tables.

    Detects tables by looking for:
    - Multiple blocks at similar Y positions (table rows)
    - Blocks with table-like content

    Args:
        blocks: List of text blocks.

    Returns:
        Filtered list with table content removed.
    """
    if not blocks:
        return blocks

    # Group blocks by page and approximate Y position (row detection)
    # Blocks within 1% of page height are considered same row
    filtered = []

    for page_num in set(b.page_num for b in blocks):
        page_blocks = [b for b in blocks if b.page_num == page_num]

        # Group by Y position (rounded to detect rows)
        y_groups: dict[float, list[TextBlock]] = {}
        for block in page_blocks:
            y_key = round(block.y_ratio, 2)  # Group within ~1% of page
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(block)

        for y_key, row_blocks in y_groups.items():
            # If many blocks at same Y position, likely a table row
            if len(row_blocks) >= 3:
                # Check if most blocks look like table cells
                table_like = sum(1 for b in row_blocks if _is_table_like_text(b.text))
                if table_like >= len(row_blocks) * 0.5:
                    # Skip this entire row - it's a table
                    continue

            # Filter individual blocks that look like table content
            for block in row_blocks:
                if not _is_table_like_text(block.text):
                    filtered.append(block)

    # Sort by page and position (top to bottom)
    filtered.sort(key=lambda b: (b.page_num, -b.y_ratio))
    return filtered


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

            # Extract characters with their font sizes
            # LTChar has font size, LTAnno is whitespace (use size=-1 to always keep)
            chars_with_sizes: list[tuple[str, float]] = []
            for line in element:
                if isinstance(line, LTTextLineHorizontal):
                    for char in line:
                        if isinstance(char, LTChar):
                            chars_with_sizes.append((char.get_text(), char.size))
                        elif isinstance(char, LTAnno):
                            # Whitespace/newlines - always keep (use -1 as marker)
                            chars_with_sizes.append((char.get_text(), -1))

            if not chars_with_sizes:
                text = element.get_text().strip()
                if text:
                    blocks.append(
                        TextBlock(
                            text=text,
                            y_ratio=element.y0 / page_height if page_height > 0 else 0.5,
                            font_size=10.0,
                            page_num=page_num,
                        )
                    )
                continue

            # Find dominant font size (most common, excluding whitespace markers)
            font_sizes = [size for _, size in chars_with_sizes if size > 0]
            if not font_sizes:
                continue
            size_counts = Counter(round(s, 1) for s in font_sizes)
            dominant_size = max(size_counts, key=lambda x: size_counts[x])

            # Filter out superscript/subscript characters (< 70% of dominant size)
            # Keep whitespace (size=-1) and normal-sized characters
            min_size = dominant_size * 0.7
            filtered_text = "".join(
                char for char, size in chars_with_sizes if size < 0 or size >= min_size
            )

            text = filtered_text.strip()
            if not text:
                continue

            # Calculate Y position as ratio (0=bottom, 1=top)
            y_ratio = element.y0 / page_height if page_height > 0 else 0.5

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

    # Filter out table content first
    blocks = _filter_table_blocks(blocks)

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

        # Skip figure/table captions
        if _is_caption(block.text):
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
    # === REMOVE ACADEMIC/PAPER ARTIFACTS ===
    # Remove inline citations like (Smith et al., 2020) or (Smith, 2020; Jones, 2019)
    # Also handles (Chen, 2018; Lee et al., 2020)
    text = re.sub(r"\([^()]*\b\d{4}[a-z]?\b[^()]*\)", "", text)

    # Remove author-year citations like "Smith (2020)" or "Smith et al. (2020)"
    text = re.sub(
        r"\b[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)?\s*\(\d{4}[a-z]?\)", "", text
    )

    # Clean up "by [Author]" patterns - remove the author part, keep "by" for grammar
    # "by Smith" -> "" (will be cleaned up), "study by Smith found" -> "study found"
    text = re.sub(
        r"\bby\s+[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)?\s*,?\s*(?=found|showed|demonstrated|reported|observed|noted|suggested|concluded|argued|claimed|stated|proposed|discovered|revealed|indicated|confirmed)",
        "",
        text,
    )

    # Remove orphaned "et al." and similar
    text = re.sub(r"\s+et\s+al\.?,?\s*", " ", text)

    # Remove figure/table references like "see Figure 1" or "(see Table 2)"
    text = re.sub(
        r"\(?see\s+(?:Figure|Fig\.?|Table|Exhibit|Chart|Graph|Appendix)\s*\d+[a-z]?\)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standalone figure/table references like "Figure 1 shows" -> "shows"
    text = re.sub(
        r"(?:Figure|Fig\.?|Table|Exhibit|Chart|Graph)\s*\d+[a-z]?\s*(?:shows?|depicts?|illustrates?|presents?|displays?|summarizes?)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove section references like "Section 2.1" or "Chapter 3" (with surrounding context)
    text = re.sub(
        r"(?:in|see|as\s+(?:shown|described|discussed)\s+in|according\s+to)\s+(?:Section|Chapter|Part)\s*\d+(?:\.\d+)*,?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?:Section|Chapter|Part)\s*\d+(?:\.\d+)*", "", text, flags=re.IGNORECASE)

    # Remove equation references like "Equation 1" or "Eq. (2)"
    text = re.sub(r"(?:Equation|Eq\.?)\s*\(?\d+\)?", "", text, flags=re.IGNORECASE)

    # Remove DOIs
    text = re.sub(r"(?:doi:|DOI:?)\s*10\.\d{4,}/[^\s]+", "", text, flags=re.IGNORECASE)

    # Remove arXiv references
    text = re.sub(r"arXiv:\d{4}\.\d{4,}(?:v\d+)?", "", text, flags=re.IGNORECASE)

    # Remove ISSN/ISBN numbers
    text = re.sub(r"(?:ISSN|ISBN)[:\s]*[\d-]+", "", text, flags=re.IGNORECASE)

    # Remove page ranges like "pp. 123-456" or "p. 42" or "pages 10-20"
    text = re.sub(r"(?:p{1,2}\.?|pages?)\s*\d+(?:\s*[-–—]\s*\d+)?", "", text, flags=re.IGNORECASE)

    # Remove volume/issue numbers like "Vol. 12, No. 3" (entire phrase)
    text = re.sub(
        r"(?:Vol(?:ume)?\.?\s*\d+,?\s*)?(?:Issue|No\.?)\s*\d+,?\s*", "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"Vol(?:ume)?\.?\s*\d+,?\s*", "", text, flags=re.IGNORECASE)

    # Remove copyright notices
    text = re.sub(r"©\s*\d{4}[^.]*\.", "", text)
    text = re.sub(r"Copyright\s*©?\s*\d{4}[^.]*\.", "", text, flags=re.IGNORECASE)

    # Remove "All rights reserved" and similar
    text = re.sub(r"All rights reserved\.?", "", text, flags=re.IGNORECASE)

    # Remove asterisks used for footnote markers
    text = re.sub(r"\*{1,3}(?=\s|$)", "", text)

    # === NORMALIZE NEWLINES FIRST ===
    # Convert various newline formats to standard \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace single newlines (mid-sentence line breaks) with spaces
    # Keep double newlines as paragraph separators
    # First, normalize multiple newlines to exactly two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace single newlines that aren't paragraph breaks with spaces
    # A single newline not preceded by sentence-ending punctuation is likely a line wrap
    text = re.sub(r"(?<![.!?:\n])\n(?!\n)", " ", text)

    # === CODE AND TECHNICAL CONTENT ===
    # Handle common programming patterns that read poorly

    # === REMOVE URLS AND TECHNICAL STRINGS FIRST ===
    # URLs (various formats) - remove completely
    text = re.sub(r"https?://[^\s<>\"')\]]+", "", text)
    text = re.sub(r"www\.[^\s<>\"')\]]+", "", text)
    text = re.sub(r"ftp://[^\s<>\"')\]]+", "", text)

    # UUIDs (with or without dashes) - must come before git hash pattern
    uuid_pattern = (
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-" r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
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

    # Remove superscript characters (often footnote references)
    # Includes Unicode superscript digits, letters, and modifier letters
    superscripts = (
        "⁰¹²³⁴⁵⁶⁷⁸⁹"  # Superscript digits
        "⁺⁻⁼⁽⁾"  # Superscript operators
        "ⁿⁱ"  # Common superscript letters
        "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"  # Superscript lowercase
        "ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂᴬᴭᴮᴯᴰᴱᴲᴳᴴᴵᴶᴷᴸᴹᴺᴻᴼᴽᴾᴿᵀᵁᵂ"  # Superscript uppercase
        "ᶦᶧᶨᶩᶪᶫᶬᶭᶮᶯᶰᶱᶲᶳᶴᶵᶶᶷᶸᶹᶺᶻᶼᶽᶾᶿ"  # More modifier letters
        "ʰʱʲʳʴʵʶʷʸʹʺʻʼʽˀˁˆˇˈˉˊˋˌˍˎˏːˑ"  # Modifier letters
    )
    for char in superscripts:
        text = text.replace(char, "")

    # Also use regex to catch any remaining superscript-like characters
    # Unicode categories for superscripts and modifiers
    text = re.sub(r"[\u2070-\u209F]", "", text)  # Superscripts and Subscripts block
    text = re.sub(r"[\u1D2C-\u1D6A]", "", text)  # Phonetic Extensions (modifier letters)
    text = re.sub(r"[\u1D78-\u1D7F]", "", text)  # More phonetic extensions
    text = re.sub(r"[\u02B0-\u02FF]", "", text)  # Spacing Modifier Letters

    # Remove subscript characters
    subscripts = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ"
    for char in subscripts:
        text = text.replace(char, "")

    # Convert smart quotes to simple quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201e", '"').replace("\u201f", '"')

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

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Ensure space after punctuation (but not before another punctuation)
    text = re.sub(r"([.,;:!?])([^\s.,;:!?'\"])", r"\1 \2", text)

    # === FINAL WHITESPACE NORMALIZATION ===
    # This must happen LAST after all substitutions that can create gaps

    # Collapse all whitespace (spaces, tabs, multiple spaces) to single space
    # Do this per-line to preserve intentional paragraph breaks
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        # Replace any sequence of whitespace with single space
        line = re.sub(r"[ \t]+", " ", line)
        # Strip leading/trailing whitespace from each line
        line = line.strip()
        normalized_lines.append(line)

    text = "\n".join(normalized_lines)

    # Remove excessive blank lines (keep max 1 blank line between paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove blank lines at start/end
    text = text.strip()

    return text
