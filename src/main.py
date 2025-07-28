import os
import json
import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# Import core modules
import config
import pdf_processor
import semantic_analyzer

# Main processing function
def run_document_intelligence(doc_paths, persona, job_to_be_done):
    """Run the full document analysis pipeline."""
    print("Starting document intelligence pipeline...")

    # Extract PDF content using multiprocessing
    with Pool(processes=min(cpu_count(), len(doc_paths))) as pool:
        results = pool.map(pdf_processor.extract_outline_from_pdf, doc_paths)

    # Combine all extracted sections
    sections = [
        {**s, "document": r["title"]}
        for r in results if r and r["outline"]
        for s in r["outline"]
    ]
    if not sections:
        print("No valid sections could be extracted. Halting analysis.")
        return None

    # Create keyword profile and find generic headings
    profile = semantic_analyzer.create_persona_profile(persona, job_to_be_done)
    generic_headings = semantic_analyzer.identify_generic_headings(sections)

    # Generate refined query for ranking
    query = f"For a {persona} focusing on '{job_to_be_done}', find content about: {', '.join(profile['positive'])}."
    print(f"Refined query for ranking: {query}")

    # Rerank sections and passages using threads
    with ThreadPoolExecutor() as executor:
        future_sections = executor.submit(semantic_analyzer.rerank_sections, sections, profile, generic_headings, query)

        # Prepare passages for ranking
        passages = [
            {"text": p.strip(), "document": s["document"], "page_number": s["page"]}
            for s in sections
            for p in s["content"].split('\n\n') if len(p.split()) > 20
        ]
        future_passages = executor.submit(semantic_analyzer.rerank_passages, passages, query)

        top_sections = future_sections.result()
        top_passages = future_passages.result()

    # Format section ranking results
    extracted_sections_output = [
        {"document": s["document"], "section_title": s["title"], "importance_rank": i+1, "page_number": s["page"]}
        for i, s in enumerate(top_sections)
    ]

    # Format passage ranking results
    subsection_analysis_output = [
        {"document": p["document"], "page_number": p["page_number"], "refined_text": p["text"]}
        for p in top_passages
    ]

    # Assemble final result
    result = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in doc_paths],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis_output
    }

    print("Analysis complete.")
    return result

# Entry point for standalone execution
if __name__ == "__main__":
    try:
        with open(config.INPUT_JSON_PATH, "r", encoding="utf-8") as f:
            app_config = json.load(f)
        persona = app_config["persona"]["role"]
        job = app_config["job_to_be_done"]["task"]
        test_case = app_config.get("challenge_info", {}).get("test_case_name", "custom")
        pdfs = [os.path.join(config.INPUT_DIR, d["filename"]) for d in app_config["documents"]]
    except Exception as e:
        print(f"Error reading or parsing input configuration: {e}")
        exit()

    print(f"--- Running Test Case: {test_case} ---")
    if pdfs:
        final_result = run_document_intelligence(pdfs, persona, job)
        if final_result:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(config.OUTPUT_DIR, f"{test_case}_output.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=4, ensure_ascii=False)
            print(f"Output successfully saved to: {out_path}")
