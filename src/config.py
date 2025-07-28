import os

# Input and output directories
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

# Path to the input JSON config
INPUT_JSON_PATH = os.path.join(INPUT_DIR, "input.json")

# Model names or local paths
EMBEDDING_MODEL_NAME = "/app/models/all-MiniLM-L6-v2"
RANKING_MODEL_NAME = "/app/models/ms-marco-MiniLM-L-6-v2"

# Similarity threshold for generic headings
GENERIC_HEADING_SIMILARITY_THRESHOLD = 0.25

# Number of top sections and passages to output
TOP_N_RESULTS = 5
