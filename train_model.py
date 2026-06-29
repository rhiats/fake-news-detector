import pandas as pd

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
