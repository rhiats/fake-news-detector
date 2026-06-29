import pandas as pd
from sklearn.model_selection import train_test_split
from config import LEAKAGE_TOKENS, MAX_FEATURES, TEST_SIZE, RANDOM_STATE, MODEL_PATH, VECTORIZER_PATH, FAKE_DATA_PATH, TRUE_DATA_PATH   
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from preprocessing import preprocess_data

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
        Train the model using the training dataset.

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

def save_model(model, vectorizer):
    """
        Save model and vectorizer for future use.

        @p: model - trained model
        @p: vectorizer - trained tfidf vectorizer
    """
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    with open(VECTORIZER_PATH, 'wb') as file:
        pickle.dump(vectorizer, file)

    return MODEL_PATH, VECTORIZER_PATH

def main():
    """
        Training pipeline
    """

    df=load_data(FAKE_DATA_PATH, TRUE_DATA_PATH)

    df = preprocess_data(df)

    train_df,test_df = split_data(df)

    model, vectorizer = train_model(train_df)

    metrics = evaluate_model(model, vectorizer, test_df)

    model_path, vectorizer_path = save_model(model, vectorizer)

    print("Training complete!")
    print(metrics)


if __name__ == "__main__":
    main()











