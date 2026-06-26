from sentence_transformers import SentenceTransformer

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model.max_seq_length = 128
        print('ml_ready')
    return model
