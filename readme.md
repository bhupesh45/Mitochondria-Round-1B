# Adobe India Hackathon 2025 - Round 1B: Persona-Driven Document Analysis

Welcome to the **"Connecting the Dots" Challenge** by Adobe India Hackathon 2025. This project addresses **Round 1B: Persona-Driven Document Intelligence**, where the goal is to make sense of documents not just by reading them, but by intelligently analyzing and ranking their content for specific personas.

---

## How to Run

### 1. Build Docker Image

```bash
$ docker build --platform linux/amd64 -t round1b:mitochondria .
```

### 2. Run Container (Hackathon-Compliant Execution)

```bash
$ docker run --rm \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    --network none \
    round1b:mitochondria
```

---

## Project Objective

You are provided with a collection of PDFs and a **persona** with a defined **job-to-be-done**. The goal is to:

* Parse and analyze the PDFs.
* Extract, classify, and rank relevant sections/subsections.
* Generate a structured JSON output tailored to the persona's needs.

---

## Project Structure

```
adobe_round_1B/
├── Dockerfile
├── requirements.txt
├── input/
│   ├── *.pdf                # Source documents
│   └── input.json           # Persona & job-to-be-done definition
├── output/
│   └── menu_planning_output.json  # Final structured output
└── src/
    ├── main.py              # Main orchestration script
    ├── config.py            # Configuration parameters
    ├── pdf_processor.py     # PDF parsing and preprocessing
    ├── semantic_analyzer.py # NLP and relevance analysis
    └── ml_models.py         # ML models for classification & ranking
```

---

## Key Features

* **Advanced Text Classification**: Identifies headings, context, and relevance.
* **Optimized for Speed**: Processes up to 5 PDFs in under 60 seconds using multiprocessing.
* **Offline-First**: Runs fully offline inside a Docker container. No external API calls.
* **Semantic Ranking**: Uses embeddings, domain-specific keywords, and heuristic rules.

---

## Constraints (Hackathon Requirements)

| Constraint        | Requirement                               |
| ----------------- | ----------------------------------------- |
| Execution Time    | ≤ 60 seconds for 3-5 PDFs                 |
| Model Size        | ≤ 1GB                                     |
| Network Access    | No internet access during execution       |
| CPU Compatibility | CPU only (amd64), Docker support required |

---

## Tech Stack

* **Language**: Python 3.13
* **Libraries**: numpy==2.3.2, pdfplumber==0.11.7, sentence\_transformers==5.0.0
* **Containerization**: Docker
* **Parallelism**: Multiprocessing & ThreadPoolExecutor

---


## Output Format

```json
{
    "metadata": {
        "input_documents": [
            "Learn Acrobat - Create and Convert_1.pdf",

            "Learn Acrobat - Fill and Sign.pdf",
            "Learn Acrobat - Generative AI_1.pdf",
            "Learn Acrobat - Generative AI_2.pdf",
            "Learn Acrobat - Request e-signatures_1.pdf",
            "Learn Acrobat - Request e-signatures_2.pdf",

        ],
        "persona": "HR professional",
        "job_to_be_done": "Create and manage fillable forms for onboarding and compliance.",
        "processing_timestamp": "2025-07-10T15:34:33.350102"
    },
    "extracted_sections": [
        {
            "document": "Learn Acrobat - Fill and Sign.pdf",
            "section_title": "Change flat forms to fillable (Acrobat Pro)",
            "importance_rank": 1,
            "page_number": 12
        },
        ......
    ],
    "subsection_analysis": [
        {
            "document": "Learn Acrobat - Fill and Sign.pdf",
            "refined_text": "To create an interactive form, use the Prepare Forms tool. See Create a form from an existing document.",
            "page_number": 12
        },
        ......
    ]
}
```

---

## Author

* Team: *Mitochondria*
* Event: Adobe India Hackathon 2025 - Round 1B
