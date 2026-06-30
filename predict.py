import pickle
from config import MODEL_PATH, VECTORIZER_PATH

def load_model():
    """
        Load the model and vectorizer.
    """
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, 'rb') as file:
        vectorizer = pickle.load(file)

    return model,vectorizer

