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

