import pickle
from config import MODEL_PATH, VECTORIZER_PATH, top_k
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

def explain_prediction(text, model, vectorizer):
    """
        Returns the top contributing words (your "Fake News Detective Explanation Layer").

        @p: text (str) - string of text that the user inputs to check veracity.
        @p: model - trained model
        @p: vectorizer - trained vectorizer

        @r: top fake and real words
    """

    #Clean text
    text = clean_text(text)

    # transform text
    X = vectorizer.transform([text])
    
    # map word → importance
    words = vectorizer.transform([text]).nonzero()[1]
    
    contributions = []
    
    for idx in words:
        word = feature_names[idx]
        weight = coeff[idx]
        contributions.append((word, weight))
    
    # sort strongest signals
    top_fake = sorted(contributions, key=lambda x: x[1], reverse=True)[:top_k]
    top_real = sorted(contributions, key=lambda x: x[1])[:top_k]
    
    return {
        "top_fake_words": top_fake,
        "top_real_words": top_real
    }
