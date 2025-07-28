import re
import numpy as np
import ml_models
import config

# Identify headings with low similarity to their content
def identify_generic_headings(sections):
    """Find headings with low semantic similarity to content."""
    print("Identifying generic headings...")
    titles = [s['title'] for s in sections]
    contents = [s['content'] for s in sections]

    # Generate embeddings for titles and contents
    title_emb = ml_models.EMBEDDING_MODEL.encode(titles, show_progress_bar=False, batch_size=16)
    content_emb = ml_models.EMBEDDING_MODEL.encode(contents, show_progress_bar=False, batch_size=16)

    # Compute cosine similarities
    similarities = np.einsum('ij,ij->i', title_emb, content_emb)

    # Collect generic headings below threshold
    generic = {sections[i]['title'].lower().strip() for i in range(len(similarities))
               if similarities[i] < config.GENERIC_HEADING_SIMILARITY_THRESHOLD}

    print(f"Generic headings found: {len(generic)}")
    return generic

# Extract basic keywords from text
def extract_keywords(text):
    """Extract keywords of 4+ letters from text."""
    return list(set(re.findall(r'\b[a-z-]{4,}\b', text.lower())))

# Create keyword profile from persona and task
def create_persona_profile(persona, job_to_be_done):
    """Generate keyword profile from persona and job."""
    print("Creating persona profile...")
    keywords = extract_keywords(persona + " " + job_to_be_done)
    print(f"Positive keywords for persona: {keywords}")
    return {"positive": keywords, "negative": []}

# Rerank sections using cross-encoder and heuristics
def rerank_sections(sections, profile, generic_headings, query):
    """Rerank sections by relevance using ML and rules."""
    print("Reranking sections for relevance...")
    pairs = [(query, s["content"]) for s in sections]
    scores = ml_models.RANKING_MODEL.predict(pairs, show_progress_bar=False)

    for i, sec in enumerate(sections):
        score = scores[i]
        title, content = sec['title'].lower(), sec['content'].lower()

        # Boost if keywords found
        if any(k in title or k in content for k in profile['positive']):
            score += 15.0
        # Penalize generic headings
        if title.strip() in generic_headings:
            score -= 10.0

        sec['final_score'] = score

    sections.sort(key=lambda x: x['final_score'], reverse=True)
    return sections[:config.TOP_N_RESULTS]

# Rerank individual passages for detailed output
def rerank_passages(passages, query):
    """Rerank passages for fine-grained answers."""
    print(f"Reranking {len(passages)} passages for relevance...")
    pairs = [(query, p["text"]) for p in passages]
    scores = ml_models.RANKING_MODEL.predict(pairs, show_progress_bar=False)

    for i, p in enumerate(passages):
        p['score'] = scores[i]

    passages.sort(key=lambda x: x['score'], reverse=True)
    return passages[:config.TOP_N_RESULTS]
