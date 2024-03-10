
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import sys
sys.path.append('/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/')
from Backend.Data.load_data import loadFileIntoDict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation

num_data=500
        
D= loadFileIntoDict("/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/ID2Genre.txt")
PATH = '/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/model_weights_multi.pth'

punctuation = list(punctuation)
def cleanDesc(df):
  sw_nltk = stopwords.words('english')
  for i,desc in enumerate(df['Description']):
    desc = [word.lower() for word in desc.split() if (word not in sw_nltk and word not in punctuation)]

    df['Description'][i]=  " ".join(desc)
  return df

def cleanGenres(df):
    cleaned_genre_col=[]
    for genre in df['Genres']:
        spaced_entries =[entry for entry in genre[1:-1].strip().split(",")]
        cleaned_genre_col.append([g.strip()[1:-1] for g in spaced_entries])

    df['Genres']= pd.Series(cleaned_genre_col)
    return df
'''def writeToFile(file, genres):
    with open(file, 'w') as f:
        for i, key in enumerate(genres.keys()):
            f.write(i + ": " + key)
    f.close()'''
    
def getMostFreqGenres(df):
  keys_to_drop = ['personal development', 'biography', 'dystopia', 'science fiction fantasy','fiction', 'memoir', 'spirituality', 'classics', 'biography memoir' , 'new adult', 'thriller', 'suspense', 'literary fiction', 'christian', 'british literature', 'paranormal', 'short stories', 'literature', 'young Adult', 'audiobook', 'novels', 'history', 'mystery thriller', 'adult' , 'chick lit', 'contemporary romance', 'contemporary', 'adult fiction', 'urban fantasy', 'middle grade', 'historical', 'american']
  freq={}
  for genre in df['Genres']:
    for g in genre:
      if g not in freq.keys():
        freq[g]=1
      else:
        freq[g] += 1
  print('old frwq', {k: v for k, v in sorted(freq.items(), key=lambda item: item[1])})
  newfreq={}
  for key,ent in freq.items():
    if  ent > 600 and key.lower() not in keys_to_drop:
      newfreq[key]=ent
  return newfreq.keys()

def store_most_frequent_genres(df, most_frequent):
  # Remove labels from entries
    df['Genres'] = df['Genres'].apply(lambda x: [item for item in x if item in most_frequent])
   # most_freq_genres =[df['Genres'][i] for i in range(len(df))]
    return df

def one_hot(df, col_name):
    col = df[col_name].tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(col)
    one_hot_encodings = mlb.transform(col)

    #store one_hot_encodings in a new col
    d = {}
    for i,classes in enumerate(mlb.classes_):
        d[i] = classes
  
    df["genre_id"]=[[0]*len(df) for i in range(len(df))]
    for i in range(len(df)):
        df["genre_id"][i]=one_hot_encodings[i]
    df ["genre_id"] = [list(map(float, target)) for target in df["genre_id"]] 
    return df      


def getDF():

    data_path = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/Backend/Data/goodreads_data.csv"
    df = pd.read_csv(data_path)


    df = df.drop(columns=['Unnamed: 0', 'Book', 'Author', 'Avg_Rating', 'Num_Ratings', 'URL'])
    df.dropna(inplace=True)
    df = df[df['Genres']!= '[]']

    df.reset_index(inplace=True, drop=True)


    df = cleanGenres(df)
    df = cleanDesc(df)
    most_freq_genres = getMostFreqGenres(df)
    df = store_most_frequent_genres(df, most_freq_genres)
    df  =one_hot(df, 'Genres')
    return df

'''def get_labels():
    data_path = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/train_data.csv"
    df = pd.read_csv(data_path)

    return df['genre'].unique().tolist()'''

df = getDF()