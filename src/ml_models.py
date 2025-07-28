from sentence_transformers import SentenceTransformer, CrossEncoder
import config

# Load embedding and ranking models
def load_models():
    """Load embedding and ranking models from given names or paths."""
    print("Loading ML models...")
    try:
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device='cpu')
        ranking_model = CrossEncoder(config.RANKING_MODEL_NAME, device='cpu')
        print("Models loaded successfully.")
        return embedding_model, ranking_model
    except Exception as e:
        print(f"Model loading failed: {e}")
        exit()

# Load models once on import
EMBEDDING_MODEL, RANKING_MODEL = load_models()
