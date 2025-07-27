import datetime
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pdfplumber

# --- ML & NLP Dependencies ---
from sentence_transformers import CrossEncoder, SentenceTransformer

# --- Paths & Configuration ---
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
INPUT_JSON_PATH = os.path.join(INPUT_DIR, "input.json")

# --- Load Models ---
print("Loading ML models...")
try:
    EMBEDDING_MODEL = SentenceTransformer("/app/models/all-MiniLM-L6-v2", device="cpu")
    RANKING_MODEL = CrossEncoder("/app/models/ms-marco-MiniLM-L-6-v2", device="cpu")
    print("Models loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load models. Error: {e}")
    exit()


def generate_ml_keywords_from_distilled_context(
    job_description, context_text, top_n=15
):
    """Generate context-specific keywords based on relevance, centrality, and prominence."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", context_text)
    context_sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    if not context_sentences:
        return []

    job_embedding = EMBEDDING_MODEL.encode([job_description])
    sentence_embeddings = EMBEDDING_MODEL.encode(context_sentences)
    similarities = np.dot(sentence_embeddings, job_embedding.T).flatten()

    top_sentence_indices = similarities.argsort()[-40:][::-1]
    distilled_context = " ".join([context_sentences[i] for i in top_sentence_indices])
    distilled_embeddings = np.array(
        [sentence_embeddings[i] for i in top_sentence_indices]
    )

    text_blob = re.sub(r"[^\w\s-]", "", distilled_context.lower())
    words = text_blob.split()

    candidates = []
    for n in range(1, 5):
        candidates += [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    stopwords = set(
        [
            "the",
            "a",
            "an",
            "and",
            "or",
            "in",
            "on",
            "for",
            "with",
            "to",
            "of",
            "is",
            "are",
            "was",
            "were",
            "it",
            "as",
            "by",
            "page",
            "chapter",
            "section",
            "document",
            "introduction",
            "conclusion",
            "summary",
            "fig",
            "figure",
        ]
    )
    valid_candidates = [
        p
        for p in set(candidates)
        if len(p) > 3
        and p not in stopwords
        and not p.isdigit()
        and all(w not in stopwords for w in p.split())
    ]
    if not valid_candidates:
        return []

    candidate_embeddings = EMBEDDING_MODEL.encode(valid_candidates)
    relevance_scores = np.dot(candidate_embeddings, job_embedding.T).flatten()
    centrality_scores = np.mean(
        np.dot(candidate_embeddings, distilled_embeddings.T), axis=1
    )
    prominence_scores = np.log1p(
        [distilled_context.lower().count(p) for p in valid_candidates]
    )

    combined_scores = (
        (0.5 * relevance_scores)
        + (0.35 * centrality_scores)
        + (0.15 * prominence_scores)
    )
    scored_candidates = list(
        zip(valid_candidates, combined_scores, candidate_embeddings)
    )
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    final_keywords = []
    final_embeddings = []
    for keyword, score, emb in scored_candidates:
        if any(keyword in k for k, _ in final_keywords):
            continue
        if final_embeddings and np.any(np.dot(final_embeddings, emb.T) > 0.85):
            continue
        final_keywords.append((keyword, score))
        final_embeddings.append(emb)
        if len(final_keywords) >= top_n:
            break

    return [k for k, _ in final_keywords]


def extract_outline_from_pdf(pdf_path):
    """Extracts headings and corresponding content from the given PDF."""
    print(f"Processing: {os.path.basename(pdf_path)}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            words = [
                word
                for page in pdf.pages
                for word in page.extract_words(extra_attrs=["size"])
            ]
            if not words:
                return {"title": os.path.basename(pdf_path), "outline": []}

            body_size = Counter(round(w["size"]) for w in words).most_common(1)[0][0]
            candidate_lines = []

            for page_idx, page in enumerate(pdf.pages):
                cropped = page.crop(
                    (0, page.height * 0.08, page.width, page.height * 0.92)
                )
                lines = cropped.extract_text_lines(layout=True, strip=False)

                for line in lines:
                    text = re.sub(r"\s+", " ", line["text"]).strip()
                    if (
                        not text
                        or len(text.split()) > 10
                        or text.endswith(":")
                        or (text.endswith(".") and len(text.split()) > 5)
                    ):
                        continue

                    if line["chars"]:
                        font_size = round(line["chars"][0]["size"])
                        font_name = line["chars"][0]["fontname"]
                        is_bold = (
                            "bold" in font_name.lower() or "bd" in font_name.lower()
                        )
                        is_all_caps = text.isupper() and len(text.split()) > 1
                        is_larger = font_size > body_size * 1.1

                        if is_larger or (is_bold and not text.islower()) or is_all_caps:
                            candidate_lines.append(
                                {"text": text, "page": page_idx, "top": line["top"]}
                            )

            candidate_lines.sort(key=lambda x: (x["page"], x["top"]))
            outline = []

            for i, heading in enumerate(candidate_lines):
                start_page, start_top = heading["page"], heading["top"]
                end_page, end_bottom = len(pdf.pages) - 1, pdf.pages[-1].height
                if i + 1 < len(candidate_lines):
                    end_page, end_bottom = (
                        candidate_lines[i + 1]["page"],
                        candidate_lines[i + 1]["top"],
                    )

                content = ""
                for page_num in range(start_page, end_page + 1):
                    page = pdf.pages[page_num]
                    crop_box = (
                        0,
                        start_top if page_num == start_page else 0,
                        page.width,
                        end_bottom if page_num == end_page else page.height,
                    )
                    text = page.crop(crop_box).extract_text(
                        x_tolerance=3, y_tolerance=3
                    )
                    if text:
                        content += text + "\n"

                if len(content.split()) > 20:
                    outline.append(
                        {
                            "title": heading["text"],
                            "page": heading["page"] + 1,
                            "content": content.strip(),
                        }
                    )

        return {"title": os.path.basename(pdf_path), "outline": outline}
    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return {"title": os.path.basename(pdf_path), "outline": []}


def identify_generic_headings(sections, threshold=0.25):
    """Identify generic headings based on low semantic similarity to their content."""
    print("Dynamically identifying generic headings...")
    titles = [s["title"] for s in sections]
    contents = [s["content"] for s in sections]

    title_embs = EMBEDDING_MODEL.encode(titles, show_progress_bar=False)
    content_embs = EMBEDDING_MODEL.encode(contents, show_progress_bar=False)
    similarities = np.array(
        [np.dot(title_embs[i], content_embs[i]) for i in range(len(sections))]
    )

    generic_headings = {
        sections[i]["title"].lower().strip()
        for i, sim in enumerate(similarities)
        if sim < threshold
    }
    print(f"Found generic headings: {generic_headings if generic_headings else 'None'}")
    return generic_headings


def generate_negative_keywords(job_description, sections, model):
    """Extract keywords from the least relevant sections to avoid in ranking."""
    print("Dynamically generating negative keywords via contrastive analysis...")
    if not sections:
        return []

    job_emb = model.encode([job_description])
    contents = [s["content"] for s in sections]
    section_embs = model.encode(contents, show_progress_bar=False)
    similarities = np.dot(section_embs, job_emb.T).flatten()

    bottom_indices = similarities.argsort()[
        : max(5, min(25, int(len(sections) * 0.15)))
    ]
    anti_context = " ".join([contents[i] for i in bottom_indices])

    words = re.sub(r"[^\w\s-]", "", anti_context.lower()).split()
    stopwords = set(
        [
            "the",
            "a",
            "an",
            "and",
            "or",
            "in",
            "on",
            "for",
            "with",
            "to",
            "of",
            "is",
            "are",
            "was",
            "were",
            "it",
            "as",
            "by",
            "we",
            "you",
            "he",
            "she",
            "they",
        ]
    )

    phrases = []
    for n in range(1, 4):
        phrases += [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    valid_phrases = [
        p for p in phrases if not any(w in stopwords for w in p.split()) and len(p) > 4
    ]
    keyword_counts = Counter(valid_phrases)
    negative_keywords = [kw for kw, _ in keyword_counts.most_common(25)]

    print(
        f"Generated negative keywords: {negative_keywords if negative_keywords else 'None'}"
    )
    return negative_keywords


def run_document_intelligence(doc_paths, persona, job_to_be_done, top_n=5):
    print("Starting document intelligence process...")
    query = f"{persona} {job_to_be_done}"

    with Pool() as pool:
        doc_results = pool.map(extract_outline_from_pdf, doc_paths)

    all_sections = [
        {**section, "document": doc_data["title"]}
        for doc_data in doc_results
        if doc_data["outline"]
        for section in doc_data["outline"]
    ]
    if not all_sections:
        print("No valid sections could be extracted.")
        return None

    generic_headings = identify_generic_headings(all_sections)
    negative_keywords = generate_negative_keywords(
        job_to_be_done, all_sections, EMBEDDING_MODEL
    )

    query_emb = EMBEDDING_MODEL.encode(query)
    contents = [s["content"] for s in all_sections]
    section_embs = EMBEDDING_MODEL.encode(contents, show_progress_bar=False)
    similarities = np.dot(section_embs, query_emb.T).flatten()

    top_indices = similarities.argsort()[-100:][::-1]
    candidates = [all_sections[i] for i in top_indices]

    context = " ".join([s["content"] for s in candidates[:30]])
    print("Distilling key phrases and entities from context...")
    keywords = generate_ml_keywords_from_distilled_context(job_to_be_done, context)
    if not keywords:
        keywords = list(set(w.lower() for w in job_to_be_done.split() if len(w) > 3))
    print(f"\nTop ML-Generated Key-Phrases: {keywords}")

    refined_query = f"For a {persona} trying to {job_to_be_done}, find specific information about these key topics: {', '.join(keywords)}."
    print(f"Refined Query: {refined_query}")

    print(f"Accurate re-ranking of top {len(candidates)} sections...")
    pairs = [(refined_query, s["content"]) for s in candidates]
    scores = RANKING_MODEL.predict(pairs, show_progress_bar=False)

    for i, s in enumerate(candidates):
        score = scores[i]
        title = s["title"].lower().strip()
        content = s["content"].lower()

        if any(k in title for k in keywords[:10]):
            score += 5.0
        if title in generic_headings:
            score += -10.0
        if any(k in content for k in negative_keywords):
            score += -20.0

        s["final_score"] = score

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top_sections = candidates[:top_n]
    extracted = [
        {
            "document": s["document"],
            "section_title": s["title"],
            "importance_rank": i + 1,
            "page_number": s["page"],
        }
        for i, s in enumerate(top_sections)
    ]

    all_passages = []
    for s in top_sections:
        passages = [
            p.strip() for p in s["content"].split("\n\n") if len(p.split()) > 20
        ]
        for p in passages:
            all_passages.append(
                {"text": p, "document": s["document"], "page_number": s["page"]}
            )

    passage_output = []
    if all_passages:
        print(f"\nRanking {len(all_passages)} passages from top sections...")
        pairs = [(refined_query, p["text"]) for p in all_passages]
        scores = RANKING_MODEL.predict(pairs, show_progress_bar=False)
        for i, p in enumerate(all_passages):
            p["score"] = scores[i]
        all_passages.sort(key=lambda x: x["score"], reverse=True)
        top_passages = all_passages[:top_n]
        passage_output = [
            {
                "document": p["document"],
                "page_number": p["page_number"],
                "refined_text": p["text"],
            }
            for p in top_passages
        ]

    result = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in doc_paths],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat(),
        },
        "extracted_sections": extracted,
        "subsection_analysis": passage_output,
    }

    print("\nAnalysis complete.")
    return result


if __name__ == "__main__":
    try:
        with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        persona = config["persona"]["role"]
        job = config["job_to_be_done"]["task"]
        test_case = config.get("challenge_info", {}).get("test_case_name", "custom")
        pdfs = [os.path.join(INPUT_DIR, doc["filename"]) for doc in config["documents"]]
    except Exception as e:
        print(f"Error reading or parsing '{INPUT_JSON_PATH}': {e}")
        exit()

    print(f"--- Loaded Test Case: {test_case} ---")

    if pdfs:
        result = run_document_intelligence(pdfs, persona, job)
        if result:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f"{test_case}_output.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"Output successfully written to: {out_path}")
