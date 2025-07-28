import os
import re
import pdfplumber
from collections import Counter

# Extract headings and content from a PDF
def extract_outline_from_pdf(pdf_path):
    """Extract structured outline from PDF using font-based heuristics."""
    print(f"Processing: {os.path.basename(pdf_path)}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract all words with font size info
            words = [word for page in pdf.pages for word in page.extract_words(extra_attrs=["size"])]
            if not words:
                return {"title": os.path.basename(pdf_path), "outline": []}

            # Get most common font size as body text size
            body_size = Counter(round(w["size"]) for w in words).most_common(1)[0][0]
            candidate_lines = []

            for page_idx, page in enumerate(pdf.pages):
                # Crop to remove headers/footers
                page_crop = page.crop((0, page.height * 0.08, page.width, page.height * 0.92))
                lines = page_crop.extract_text_lines(layout=True, strip=False)

                for line in lines:
                    text = re.sub(r'\s+', ' ', line['text']).strip()
                    # Skip lines unlikely to be headings
                    if (not text or len(text.split()) > 10 or text.endswith(':') or
                        (text.endswith('.') and len(text.split()) > 5)):
                        continue

                    if line['chars']:
                        # Analyze font to detect heading
                        line_size = round(line["chars"][0]["size"])
                        font_name = line["chars"][0]["fontname"]
                        is_bold = 'bold' in font_name.lower() or 'bd' in font_name.lower()
                        is_all_caps = text.isupper() and len(text.split()) > 1
                        is_larger = line_size > body_size * 1.1

                        if is_larger or (is_bold and not text.islower()) or is_all_caps:
                            candidate_lines.append({"text": text, "page": page_idx, "top": line["top"]})

            # Sort headings by page and position
            candidate_lines.sort(key=lambda x: (x["page"], x["top"]))
            outline = []

            for i, heading in enumerate(candidate_lines):
                # Set content start and end based on heading positions
                start_page_idx, start_top = heading["page"], heading["top"]
                end_page_idx, end_bottom = (len(pdf.pages) - 1, pdf.pages[-1].height)
                if i + 1 < len(candidate_lines):
                    next_heading = candidate_lines[i + 1]
                    end_page_idx, end_bottom = next_heading["page"], next_heading["top"]

                content = ""
                for page_num in range(start_page_idx, end_page_idx + 1):
                    page = pdf.pages[page_num]
                    # Define content crop box
                    crop_box = (0, start_top if page_num == start_page_idx else 0,
                                page.width, end_bottom if page_num == end_page_idx else page.height)
                    extracted_text = page.crop(crop_box).extract_text(x_tolerance=3, y_tolerance=3)
                    if extracted_text:
                        content += extracted_text + "\n"

                # Add section if content is substantial
                if len(content.split()) > 20:
                    outline.append({
                        "title": heading["text"],
                        "page": heading["page"] + 1,
                        "content": content.strip()
                    })

        return {"title": os.path.basename(pdf_path), "outline": outline}
    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return {"title": os.path.basename(pdf_path), "outline": []}
