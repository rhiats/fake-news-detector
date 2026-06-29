import pandas as pd
from sklearn.model_selection import train_test_split

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
    leakage_tokens = [
        "reuters",
        "washington",
        "breakingnews",
        "ap",
        "associatedpress",
    ]

    for token in leakage_tokens:
        text = text.replace(token, "")

    # collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def pre_process_data(data_df,clean_text):
    """
        Pre process the text in the dataset:
            - lowercase
            - remore hyperlinks
            - only consider lowercase letters 
            - remove newssource artifacts

        @p: data_df(dataframe) - full data
        @p: clean_text - function to clean the text in the dataset

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
    test_size=0.2, 
    stratify=data_df['label'],
    random_state=42)

    return train_df,test_df

