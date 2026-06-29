from config import LEAKAGE_TOKENS
import re

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
