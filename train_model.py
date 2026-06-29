import pandas as pd
from sklearn.model_selection import train_test_split
from config import LEAKAGE_TOKENS, MAX_FEATURES, TEST_SIZE, RANDOM_STATE
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


def load_data(fake_path, true_path):
    """
        Load dataset from csv files, add labels, and merge into one dataset

        @p: fake_path: name of fake dataset
        @p: true_path: name of true dataset

        @r: dataframe with true and false dataset merged together
    """
    fake_df= pd.read_csv(fake_path)
    true_df=pd.read_csv(true_path)

    #Add labels
    fake_df['label'] = 1
    true_df['label'] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)

    return df

def clean_text(text):
    """
        Pre process text:
            - lowercase
            - remore hyperlinks
            - only consider lowercase letters 
            - remove newssource artifacts

        @p: text (str)

        @r: text (str) cleaned text

    """

    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove numbers + punctuation
    text = re.sub(r"[^a-z\s]", "", text)

    # remove known news source artifacts
    for token in LEAKAGE_TOKENS:
        text = text.replace(token, "")

    # collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_data(data_df):
    """
        Pre process the text in the dataset:
            - lowercase
            - remore hyperlinks
            - only consider lowercase letters 
            - remove newssource artifacts

        @p: data_df(dataframe) - full data

        @r: data_df(dataframe) - includes cleaned text

    """

    data_df["clean_text"] = data_df["title"] + " " + data_df["text"]
    data_df["clean_text"] = data_df["clean_text"].apply(clean_text)

    return data_df



def split_data(data_df):
    """
        Train Test split into training and test file

        @p: data_df(dataframe) - full data
        @r: train set (dataframe), test set (dataframe) 
    """

    train_df, test_df = train_test_split(
    data_df, 
    test_size=TEST_SIZE, 
    stratify=data_df['label'],
    random_state=RANDOM_STATE)

    return train_df,test_df

def train_model(train_df):
    """
        Train the model using the training and test dataset.

        @p: train_df (dataframe)
        @p: test_df (dataframe)

        @r: model - trained model
        @r: vectorizer - tfidf vectorizer
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES
    )

    X_train = vectorizer.fit_transform(train_df["clean_text"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_df["label"])
        
    return model, vectorizer

def evaluate_model(model, vectorizer, test_df):
    """
        Evaluate the model performance.

        @p model: trained model
        @ vectorizer: trained vectorizer
        @test_df: dataframe for testing

        @r: dictorionary of training metrics
    """

    X_test = vectorizer.transform(test_df["clean_text"])
    y_pred = model.predict(X_test)

    y_true = test_df["label"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

    return metrics

