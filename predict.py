import pickle
from config import MODEL_PATH, VECTORIZER_PATH
from preprocessing import clean_text

def load_model():
    """
        Load the model and vectorizer.
    """
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, 'rb') as file:
        vectorizer = pickle.load(file)

    return model,vectorizer

def predict_article(text, model, vectorizer):
    """
        Cleans the article, vectorizes it, predicts fake or real, returns the prediction.

        @p: text (str) - string of text that the user inputs to check veracity.
        @p: model - trained model
        @p: vectorizer - trained vectorizer

        @r: pred (str) - True/Fake News prediction of text
    """

    #Cleans the article.
    text = clean_text(text)
    #Vectorizes it.
    text_vect = vectorizer.transform(text)
    #Predicts fake or real.
    pred = model.predict(text_vect)
    #Returns the prediction.

    return pred

def predict_probability(text, model, vectorizer):
    """
        Returns the probability of fake and real using predict_proba()

        @p: text (str) - string of text that the user inputs to check veracity.
        @p: model - trained model
        @p: vectorizer - trained vectorizer

        @r: prob_dict - dictionary with probability fake/real news
    """
        #Clean text
        text = clean_text(text)

        # transform text
        X = vectorizer.transform(text)
        
        # get probability
        prob = model.predict_proba(X)[0]

        prob_dict = {"prob_fake": prob[1],
        "prob_real": prob[0]
        }

        return prob_dict

