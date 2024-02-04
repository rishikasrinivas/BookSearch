import pandas as pd
import numpy as np

def getDF():
    
    data_path = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/train_data.csv"
    df = pd.read_csv(data_path)

  
    
    df = df.drop(columns=['index', 'title'])
    df.dropna(inplace=True)

    df['genre'].value_counts()
    genre2id = {genre: i for i, genre in enumerate(df['genre'].unique())}
    
    #genre = pd.get_dummies(df['genre']).astype('int64')
 
    df["genre_id"] = df['genre'].apply(lambda a: genre2id[a])
    return df
def id2genre(df):
    return {i: genre for i, genre in enumerate(df['genre'].unique())}
def get_labels():
    data_path = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/train_data.csv"
    df = pd.read_csv(data_path)

    return df['genre'].unique().tolist()

df = getDF()
print(id2genre(df))