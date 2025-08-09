#from src.services.llm.llm import client
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Tuple, Literal, Optional
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import fitz
from src.services.logging.logger import logger
import tempfile
import asyncio
import json
from collections import Counter, defaultdict
import pdfplumber
import matplotlib.pyplot as plt
import matplotlib.patches  as patches
import io
import tabulate
from IPython.display import display
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)

path_test = '/Users/saraevsviatoslav/Documents/playground/data/processed/cleaned.pdf'


prompt = """You are given the text content of a single PDF page. Your task is to decide whether this page starts a new subheading / new logical section, or whether its first narrative paragraph is a continuation of the previous page.

            Important constraints
               •	Ignore images, figures, charts, and tables. If the page opens with a table (or figure) and the first textual paragraph after it continues the previous page’s paragraph, treat it as a continuation.
                •	Ignore running headers/footers, page numbers, and decorative elements.
                •	Do not rely on rendering or fonts; base your judgment only on the provided text and simple textual cues.
                •	Do not output any explanation—output only a single word: TRUE or FALSE.

            Heuristics to use
            Consider the page starts a new subheading / section (answer TRUE) if one or more of the following holds for the first textual block (after skipping tables/figures and headers/footers):
                •	It begins with a clear subheading / title cue (e.g., a line that looks like a heading: short phrase, Title Case / ALL CAPS, numbered like “1.”, “1.2”, “A.”, “Appendix A”, “Conclusion”, “Methods”, etc.), and is then followed by a new paragraph.
                •	It starts a new topic (e.g., a definition, a new chapter/section label, or a bullet/numbered list that clearly begins a new section rather than continuing a sentence).
                •	The first paragraph clearly starts at sentence boundary with context indicating a new topic (e.g., opens with “In this section…”, “We now examine…”, “Conclusion”, “Discussion”, etc.).

            Consider the page continues the previous page (answer FALSE) if one or more holds:
                •	The first textual paragraph starts mid-sentence (e.g., begins with a lowercase word that grammatically continues a sentence, or starts with punctuation like “,” “;” “)”).
                •	The first textual paragraph references immediate continuation (“continued”, “cont.”) or resumes a list/table description already in progress.

            Ambiguity rule
                •	If evidence is insufficient or ambiguous, default to FALSE (do not split). Prefer avoiding over-splitting.

            Input
                •	CURRENT_PAGE_PDF (required): file in format pdf.

            Your task
                1.	Virtually skip headers/footers and any tables/figures in CURRENT_PAGE_TEXT.
                2.	Identify the first textual paragraph.
                3.	Decide if that paragraph marks a new subheading/section per the heuristics above.
                4.	Output only TRUE or FALSE.

            Output format
                •	Exactly one token: TRUE or FALSE (uppercase). No punctuation, no quotes, no extra text.

            Example I/O (for your internal guidance only; do not repeat in output)
                •	If the first text block is “CONCLISION AND DISCUSSION” followed by a paragraph → TRUE.
                •	If the page starts with a table, then the next paragraph begins “and were then compared across…” → FALSE.
                •	If first text starts with “In this section, we discuss…” → TRUE.
                •	If first text starts with “, which shows that …” → FALSE.

            Now produce the decision. Output only TRUE or FALSE."""


async def ask_gpt_starts_page_with_title(
    path: str | Path,
    pages_to_check: List[int],
    prompt: str,
    client: AsyncOpenAI,
    concurrency: int = 5,
) -> Dict[int, bool]:
    """
    Asynchronously checks whether given PDF pages start with a new section/subheading
    or are a continuation of the previous paragraph.

    This function:
    1. Extracts each target page as a single-page PDF (in memory, not on disk).
    2. Uploads the PDF page to OpenAI's file API.
    3. Sends the uploaded file and a text prompt to a Chat Completion model (e.g., GPT-5).
    4. Interprets the model's response as a strict TRUE/FALSE boolean.

    Args:
        path (str | Path): Path to the source PDF file.
        pages_to_check (List[int]): List of page indices (0-based) to check.
        prompt (str): The text prompt instructing the model to return only TRUE or FALSE.
        client (AsyncOpenAI): An initialized asynchronous OpenAI API client.
        concurrency (int): Maximum number of concurrent API requests to run in parallel.

    Returns:
        Dict[int, bool]: A dictionary mapping each checked page index to a boolean:
                         True  -> Page starts with a new section/subheading
                         False -> Page continues from the previous paragraph
    """
    doc = fitz.open(str(path))  # Open the PDF
    # Keep only valid page indices, sorted and unique
    pages = sorted(set(p for p in pages_to_check if 0 <= p < len(doc)))
    # Semaphore to limit concurrent requests (avoid hitting rate limits)
    sem = asyncio.Semaphore(concurrency)

    async def _process_page(p: int) -> tuple[int, bool]:
        """
        Processes a single page: extracts it, uploads it, sends it to GPT, and parses the result.
        """
        # Create a new PDF document containing only this page
        one = fitz.open()
        one.insert_pdf(doc, from_page=p, to_page=p)

        # Save it to an in-memory bytes buffer
        buf = io.BytesIO()
        one.save(buf)
        one.close()
        buf.seek(0)
        setattr(buf, "name", f"page_{p}.pdf")  # Give a name for compatibility with some APIs

        async with sem:
            # Upload page to OpenAI as a file
            uploaded = await client.files.create(
                file=buf,
                purpose="user_data",
            )

            # Send chat completion request with file + text prompt
            completion = await client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "file", "file": {"file_id": uploaded.id}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

        # Extract and normalize model's response
        raw = (completion.choices[0].message.content or "").strip().upper()
        is_new_section = raw == "TRUE"

        return p, is_new_section

    try:
        # Process all target pages in parallel
        results = await asyncio.gather(*[_process_page(p) for p in pages])
        return dict(results)
    finally:
        doc.close()
            

def define_size_of_text_and_cover(doc: fitz.Document) -> tuple[int, bool]: # -> Most common size and if first page is cover

    """
    Analyzes the font sizes used in a PDF file and determines the most common font size,
    as well as whether the most common size on the first page is larger than in the whole document.

    Args:
        file: Path to the PDF file (can be a string or Path-like object).

    Returns:
        tyle:
            - int: The most common font size across the entire document.
            - bool: True if the most common font size on the first page is larger than the overall one, False otherwise.
    """

    all_sizes = []
    sizes_first_page = []
    
    for page_index, page in enumerate(doc):
        blocks = page.get_text('dict')['blocks']

        for block in blocks:
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        size = span['size']
                        if page_index == 0:
                            sizes_first_page.append(size)
                        all_sizes.append(size)

    counter_first = Counter(sizes_first_page)
    most_common_num_on_first_page, _ = counter_first.most_common(1)[0]
    counter = Counter(all_sizes)
    most_common_num, _ = counter.most_common(1)[0]

    return most_common_num, most_common_num_on_first_page > most_common_num


def is_toc_page(page):
    blocks = page.get_text('dict')['blocks']
    count_with_number = 0
    total_lines = 0

    for block in blocks:
        if block['type'] == 0:
            for line in block.get('lines', []):
                line_text = ''.join(span['text'] for span in line['spans']).strip()
                if len(line_text) < 4:
                    continue

                total_lines +=1
                if line_text[-1].isdigit():
                    count_with_number +=1

    if total_lines == 0:
        return False
    
    return count_with_number / total_lines > 0.7


def drop_cover_and_table_of_contents(doc: fitz.Document):
    """
    Removes the cover page and table of contents pages from a PDF document.

    The function checks if the first page is likely a cover (based on font size or layout)
    and deletes it if so. It then scans subsequent pages to detect and remove pages 
    that appear to be part of the table of contents.

    Args:
        doc (fitz.Document): The PDF document to process (mutated in place).

    Returns:
        fitz.Document: The same document with the cover and table of contents pages removed.
    """

    _, first_page_is_cover = define_size_of_text_and_cover(doc)

    if first_page_is_cover:
        doc.delete_page(0)

    # Find the pages of table of content
    toc_pages = []
    for page_num in range(1, doc.page_count):
            page = doc.load_page(page_num)

            if is_toc_page(page):
                toc_pages.append(page_num)
    if toc_pages:
        doc.delete_pages(toc_pages)

    return doc


def find_repeated_headers_and_footers_by_position(
    doc_cleaned: fitz.Document,
    top_ratio: float = 0.05,
    bottom_ratio: float = 0.05,
    min_repeats: float = 0.5,
    round_precision: int = 1
) -> Dict[str, Any]:
    """
    Detects likely headers and footers in a PDF based on their repeated position
    near the top or bottom of the page, regardless of their text content.

    Args:
        doc (fitz.Document): The PDF document to analys
        top_ratio (float): Portion of the top area of the page (0.1 = top 10%) to consider for header detection.
        bottom_ratio (float): Portion of the bottom area of the page to consider for footer detection.
        min_repeats (float): Minimum proportion of pages a position must appear on to be considered repeated.
        round_precision (float): Decimal precision to round Y-coordinates when groying block positions.

    Returns:
        dict: {
            "likely_header_positions": List[float],  # Y-coordinates of likely headers
            "likely_footer_positions": List[float],  # Y-coordinates of likely footers
            "position_texts": Dict[float, List[str]]  # (optional) block texts for inspection/debugging
        }
    """

    num_pages = len(doc_cleaned)

    header_positions = Counter()
    footer_positions = Counter()
    position_texts = defaultdict(list)  # useful for debugging

    for page_index, page in enumerate(doc_cleaned):
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue  # skip image blocks, empty blocks, etc.

            y0 = block["bbox"][1]
            y1 = block["bbox"][3]

            # Combine all text from the block
            text = "".join(
                span["text"] for line in block["lines"] for span in line["spans"]
            ).strip()

            if not text:
                continue

            # Check if the block is near the top of the page
            if y0 < page_height * top_ratio:
                y_key = round(y0, round_precision)
                header_positions[y_key] += 1
                position_texts[y_key].append(text)

            # Check if the block is near the bottom of the page
            elif y1 > page_height * (1 - bottom_ratio):
                y_key = round(y1, round_precision)
                footer_positions[y_key] += 1
                position_texts[y_key].append(text)

    # Select positions that occur on at least `min_repeats` percent of pages
    likely_header_positions = [
        y for y, count in header_positions.items() if count / num_pages >= min_repeats
    ]
    likely_footer_positions = [
        y for y, count in footer_positions.items() if count / num_pages >= min_repeats
    ]

    return {
        "likely_header_positions": likely_header_positions,
        "likely_footer_positions": likely_footer_positions,
        "position_texts": position_texts  # optional, can be removed
    }

# =====================================================================================

docs = list(RAW_DATA_DIR.glob('*.pdf'))
for d in docs:
    doc_fitz = fitz.open(d)
    doc_cleaned = drop_cover_and_table_of_contents(doc_fitz)    
    head_foot_position = find_repeated_headers_and_footers_by_position(doc_cleaned)
    break

size_of_text, _ = define_size_of_text_and_cover(doc_cleaned)
logger.info(f"!!! General size of text in pdf: {size_of_text}")
out_path_cleaned = PROCESSED_DATA_DIR / 'cleaned.pdf'
doc_cleaned.save(out_path_cleaned)

# =====================================================================================



async def pages_starting_with_subheading(
                                    path: str | Path,
                                    doc_cleaned: fitz.Document,
                                    head_foot_position: Dict, 
                                    size_of_text: int
                                    ) -> List[int]:

    """
    Identifies pages that likely start with a subheading or section title.

    The function analyzes each page of a cleaned PDF (with cover and TOC removed),
    compares the font size of the first visible line on the page to the general body text size,
    and excludes blocks detected as headers or footers based on stable vertical position.

    Args:
        doc (fitz.Document): The original PDF document.

    Returns:
        List[int]: A list of page indices (starting from 0) that begin with a likely subheading.
    """

    pages_open_with_sub = []   # Pages that likely start a new section based on title font size
    is_table_on_page,  = is_table(path_test, head_foot_position, doc_cleaned)
     
 
    # Save doc to check the work of this function
    # out_path = RAW_DATA_DIR / f"pdf_without_garbage.pdf"
    # doc.save(out_path)
    # logger.info(f"✅ Saved demo pdf to {out_path}")


    header_y_positions = head_foot_position["likely_header_positions"]
    footer_y_positions = head_foot_position["likely_footer_positions"]
    text_found_as_footer = head_foot_position["position_texts"]

    # Take a size of first line on the page and find a number of pages opening with title
    for page_index, page in enumerate(doc_cleaned):
        sizes_on_first_line = [] 
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            y0 = block["bbox"][1]
            y1 = block["bbox"][3]

            if round(y0, 1) in header_y_positions or round(y1, 1) in footer_y_positions:
                continue  # skip header/footer

            lines = block.get('lines', [])
            for line in lines:
                line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                if line_text:  # if not just spaces
                    for span in line.get("spans", []):
                        if span["text"].strip():  # make sure this specific span isn't empty or just space
                            sizes_on_first_line.append(span["size"])
                    break  # break inner loop (line found)
            if sizes_on_first_line:
                break  # break outer loop (block found)
        if sizes_on_first_line:
            first_line_counter = Counter(sizes_on_first_line)
            most_common_size_on_first_line, _ = first_line_counter.most_common(1)[0]
            #logger.info(f"The most common size on the first line: {most_common_size_on_first_line} while the general size is {size_of_text}")

            if most_common_size_on_first_line > size_of_text: # Probably it is a title
                pages_open_with_sub.append(page_index) 

    # check if page to split not starts with table

    pages_to_check = []
    for i in pages_open_with_sub:
        
        results_for_page = [
            is_table_on_page["pdf_plumber_found"][i],
            is_table_on_page["pages_with_lines"][i],
            is_table_on_page["by_algorithm"][i]
        ]
        if any(results_for_page):
            pages_to_check.append(i)
    print(f'Pages start with head: \n{pages_open_with_sub}')
    # Ask ChatGPT about the pages we're unsure about
    checked = await ask_gpt_starts_page_with_title(path, pages_to_check, prompt, client)
    print(f"What say gpt aboun pages we're unsure: \n{checked}")
    result_list_of_pages_with_title = [p for p in pages_open_with_sub if checked.get(p, True)]
    print(f'Result points to split: \n{result_list_of_pages_with_title}')
 
    return result_list_of_pages_with_title


def text_blocks_visual(
    doc_cleaned: fitz.Document,
    num_page: int = 0,
    dpi: int = 144
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Render a PDF page to an image and overlay rectangles for each text span's bounding box.

    This is useful for visually debugging text extraction: you see the page bitmap
    and red boxes where PyMuPDF reports text spans.

    Args:
        doc_cleaned (fitz.Document): The (already cleaned) PDF document.
        num_page (int): Zero-based page index to visualize. Defaults to 0.
        dpi (int): Rasterization DPI for the page image. Bounding boxes are scaled
            accordingly (bbox coords are in PDF points, i.e., 72 DPI). Defaults to 144.

    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes): The created figure and axes.

    Notes:
        - PDF coordinates are in points (1 pt = 1/72"). If you render at `dpi != 72`,
          bounding boxes must be scaled by `dpi / 72` to align with the raster image.
        - Spans are drawn in red with 1px linewidth.
        - The y-axis is flipped to match image coordinates (origin at top-left).

    Example:
        >>> import fitz
        >>> doc = fitz.open("file.pdf")
        >>> fig, ax = text_blocks_visual(doc, num_page=2, dpi=144)
        >>> plt.show()
    """
    if doc_cleaned is None or not isinstance(doc_cleaned, fitz.Document):
        raise ValueError("doc_cleaned must be a valid fitz.Document")

    if not (0 <= num_page < len(doc_cleaned)):
        raise IndexError(f"Page index {num_page} out of range (0..{len(doc_cleaned)-1})")

    page = doc_cleaned[num_page]
    pix = page.get_pixmap(dpi=dpi)
    img = pix.tobytes("png")

    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(plt.imread(io.BytesIO(img)))

    # Scale bbox (PDF points) to rendered image pixels
    scale = dpi / 72.0

    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line.get("spans", []):
                x0, y0, x1, y1 = span["bbox"]
                # scale to match the rasterized image
                rx0, ry0, rx1, ry1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
                rect = patches.Rectangle(
                    (rx0, ry0),
                    rx1 - rx0,
                    ry1 - ry0,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none"
                )
                ax.add_patch(rect)

    # Align axes with image space (origin top-left)
    ax.set_xlim([0, pix.width])
    ax.set_ylim([pix.height, 0])
    ax.axis("off")
    plt.tight_layout()
    return fig, ax



def detect_table_like_structure(
    lines: List[List[Tuple[float, float]]],
    tolerance: float = 2.0,
    min_spans_per_line: int = 3,
    min_consecutive_matches: int = 3,
    min_match_ratio: float = 0.7,
    ratio_denominator: Literal["min","max"] = "max",
) -> Tuple[bool, List[int]]:
    """
    Detect whether there is a run of lines that share a repeating horizontal structure (table-like).

    Args:
        lines: Each line is a list of (x0, x1) span positions sorted by x0.
        tolerance: Max allowed deviation in x0/x1 to treat spans as the same column.
        min_spans_per_line: Lines with fewer spans are ignored and break the current run.
        min_consecutive_matches: Required length of a consecutive run of matching lines.
        min_match_ratio: Minimum ratio of matched spans (0..1) to call two lines similar.
        ratio_denominator: Use "max" (safer) or "min" as denominator when computing the ratio.

    Returns:
        (is_table, matched_indices)
        - is_table: True if a run was found.
        - matched_indices: indices of lines participating in the final matching run.
    """
    matched_indices: List[int] = []
    repeat_count = 0
    prev_line: Optional[List[Tuple[float, float]]] = None
    run_start_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        # Skip too-short lines; break any ongoing run
        if len(line) < min_spans_per_line:
            prev_line = None
            repeat_count = 0
            matched_indices.clear()
            run_start_idx = None
            continue

        if prev_line is None:
            prev_line = line
            run_start_idx = idx  # potential start of a run
            continue

        # Greedy one-to-one matching with "used" flags to avoid double counting
        used_prev = [False] * len(prev_line)
        matches = 0
        for x0, x1 in line:
            # find a prev span that hasn't been used and is within tolerance
            for j, (px0, px1) in enumerate(prev_line):
                if used_prev[j]:
                    continue
                if abs(x0 - px0) <= tolerance or abs(x1 - px1) <= tolerance:
                    used_prev[j] = True
                    matches += 1
                    break

        denom = max(len(line), len(prev_line)) if ratio_denominator == "max" else min(len(line), len(prev_line))
        match_ratio = matches / denom if denom > 0 else 0.0

        if match_ratio >= min_match_ratio:
            repeat_count += 1
            # include previous line once, then current
            if not matched_indices:
                # add prev index first time we confirm a match
                matched_indices.append(idx - 1)
            matched_indices.append(idx)

            if repeat_count >= (min_consecutive_matches - 1):
                return True, matched_indices
        else:
            # reset run, start over from this line
            repeat_count = 0
            matched_indices.clear()
            run_start_idx = idx

        prev_line = line

    return False, []
    


def is_table(
        path: Path | str,
        head_foot_position: Dict,
        doc_cleaned: fitz.Document) -> tuple[Dict[str, Dict[int, bool]], float]:
    
    """
    Detects presence of tables on each page of a PDF document using multiple approaches.

    This function applies three detection methods:
        1. **pdfplumber** — detects tables by parsing PDF structure.
        2. **Line detection** — checks if a page contains multiple drawn lines.
        3. **Algorithmic detection** — analyzes text alignment and spacing to detect table-like structures.

    Args:
        path (Path | str): Path to the PDF file.
        head_foot_position (dict): Dictionary containing likely header/footer Y positions.
                                   Must include key "likely_header_positions" with a list of Y coordinates.
        doc_cleaned (fitz.Document): Cleaned PDF document object from PyMuPDF.

    Returns:
        dict: A dictionary with three keys:
            - `"pdf_plumber_found"` (dict[int, bool]): Pages where `pdfplumber` detected tables.
            - `"pages_with_lines"` (dict[int, bool]): Pages containing multiple drawn lines.
            - `"by_algorithm"` (dict[int, bool]): Pages where algorithmic text-position analysis detected a table.

    Example:
        >>> results = is_table("sample.pdf", {"likely_header_positions": [50, 800]}, doc)
        >>> results["pdf_plumber_found"][0]
        True
    """
    
    pdf_plumber_found = {} # Indexes of pages with tables founded by pdfplumber
    page_founded_by_algorithm = {}

    # TODO Try to find pages with table by Pdfplumber

    with pdfplumber.open(path) as pdf:
        all_images_position = {}
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                pdf_plumber_found[i] = True # Collect index of pages
            else:
                pdf_plumber_found[i] = False # Collect index of pages

            # Find top image
            if page.images:
                start_position_img = min(img["top"] for img in page.images)
                all_images_position[i] = start_position_img
            else:
                all_images_position[i] = None

            


    # TODO Try to find tables by lines

    header_y_positions = head_foot_position["likely_header_positions"]

    pages_with_lines = {}
    for page_index, page in enumerate(doc_cleaned):
        all_lines_span = []
        line = page.get_drawings()
        if line:
            if len(line) > 1:
                pages_with_lines[page_index] = True # Lines
            else:
                pages_with_lines[page_index] = False
        else:
                pages_with_lines[page_index] = False # Lines


    # TODO Try to find tables by text position

        tolerance = 2 
        text_dict = page.get_text("dict")
        full_line = defaultdict(list)

        for block in text_dict["blocks"]:

            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span['text'].strip():
                        y = round(span["bbox"][1] / tolerance) * tolerance

                        if y in header_y_positions:
                            continue

                        full_line[y].append((
                                                round(span['bbox'][0], tolerance),
                                                round(span['bbox'][2], tolerance),
                                            ))
                        
        for y, spans in sorted(full_line.items()):
            spans.sort(key=lambda s: s[0])
            all_lines_span.append(spans)

        is_table, _ = detect_table_like_structure(all_lines_span)
        if is_table:
            page_founded_by_algorithm[page_index] = True
        else:
            page_founded_by_algorithm[page_index] = False

    signs = {'pdf_plumber_found': pdf_plumber_found,
                'pages_with_lines': pages_with_lines,
                'by_algorithm': page_founded_by_algorithm}
    
    return signs, start_position_img



async def split_pdf_to_temp_files(
        path: str | Path, 
        doc: fitz.Document, 
        head_foot_position: Dict[str, Any], 
        size_of_text: int
        ) -> List[Path]:
    
    """
    Splits a PDF document into separate sections based on subheading positions,
    saves each part as a temporary PDF file, and returns their file paths.

    The function detects pages that likely start a new section (using `pages_starting_with_subheading`)
    and splits the original PDF into smaller parts. Each part is saved to a temporary file.

    Args:
        doc (fitz.Document): The PDF document to split.

    Returns:
        List[Path]: A list of file paths pointing to temporary PDF files.
                    These can be passed directly to OpenAI file yload.
    """
    split_points = await pages_starting_with_subheading(path, doc, head_foot_position, size_of_text)
    split_points = sorted(split_points)
        

    if 0 not in split_points:
        split_points.insert(0, 0)

    if len(doc) not in split_points:
        split_points.append(len(doc))  # Ensure we cover till the end

    saved_paths = []
    base_name = Path(doc.name).stem if doc.name else "document"

    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]

        new_doc = fitz.open()
        for page_num in range(start, end):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        file_path = RAW_DATA_DIR / f"{base_name}_part_{i + 1}.pdf"
        new_doc.save(file_path)
        saved_paths.append(file_path)

    return saved_paths


async def test_saving_pdf(path_pdf: Path = RAW_DATA_DIR):
    pdfs = list(path_pdf.glob('*.pdf'))

    if not pdfs:
        print('PDFs not found')
        return

    for pdf in pdfs:
        await split_pdf_to_temp_files(
            path=pdf,
            doc=doc_cleaned,
            head_foot_position=head_foot_position,
            size_of_text=size_of_text
            )
    
def test_is_table(path: str = '/Users/saraevsviatoslav/Documents/playground/data/processed/cleaned.pdf'):
    is_table(path, head_foot_position, doc_cleaned)

def test_text_block_visual():
    fig, ax = text_blocks_visual(doc_cleaned, 10)
    plt.show()

def test_pages_starting_with_subheading():
    pages_to_check = pages_starting_with_subheading(doc_cleaned, head_foot_position, size_of_text)
    return pages_to_check
async def test_ask_gpt_starts_page_with_title():
    pages_to_check = test_pages_starting_with_subheading()
    result = await ask_gpt_starts_page_with_title(path_test, pages_to_check, prompt, client)
    print(result)

