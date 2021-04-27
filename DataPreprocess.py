import pandas as pd
import string
import re
import numpy as np
import math
import random
import regex
import torch
from torchtext.legacy import data
from torchtext.legacy.data import  Dataset, Example

def clean_data(df):

  df = df.apply(lambda x:' '.join(x.lower() for x in x.split())) #remove english words
  df = df.apply(lambda x: regex.sub(r'(#[^\s]*)*', '',x))                                    #removing hashtags     
  df = df.apply(lambda x: regex.sub(r'(@[\w]*)*[\d~\|\p{Punct}*]*(http[^\s]*)*', '',x)) 
  df = df.apply(lambda x: regex.sub(r'<[^<]+?>','',x)) #remove html 
  df = df.apply(lambda x: regex.sub(r'href=','',x)) 
  df = df.apply(lambda x: x.lower()) 
  df = df.apply(lambda x: remove_emoji(x))
  return df
                                                           #make lower case
#remove emojis, this script was taken from github
def remove_emoji(text):
    emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags 
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hindi_stopwords(df):
   #import and remove stopwords
  stopurl = 'Data/stopwords-hi.txt'
  stopwords = pd.read_csv(stopurl, sep='\t', header=None)
  df = df.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords[:][0].tolist()))
  return df

def remove_bengali_stopwords(df):
  stopurl = 'Data/stopwords-bn.txt'
  stopwords = pd.read_csv(stopurl, sep='\t', header=None)
  df = df.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords[:][0].tolist()))
  return df  

def reduce_bengali(df):
#Making roughly equal to Hindi set which has 2469 hate  and 2196 non hate
#lets just make 4700 Bengali corpus, 2500 hate and 2200 not hate
  not_hate = df.loc[10001:12200].copy()
  df = df.loc[1:2500].copy()
  df = df.append(not_hate)
  df = df.sample(frac=1) #shuffle it so its random
  return df

def hindi_drop_columns(df): #we need to drop uneccesary columns before converting into torchtext objects
  df = df.drop(columns=['text_id', 'task_2','task_3'])
  return df

def bengali_drop_columns(df): #we need to drop uneccesary columns before converting into torchtext objects
  df = df.drop(columns=[2])
  return df

def rename_df_columns(df):
  df.columns = ['text','label']
  return df


#convert from pandas to torchtext dataframes
'''
DEAR TA'S: 
THE CODE IN THIS CELL WAS TAKEN FROM STACK OVERFLOW AND MODIFIED SLIGHTLY
BECAUSE WE WERE HAVING TROUBLE CONVERTING OUR CLEANED PANDA DATAFRAME 
INTO A SHAPE TORCHTEXT COULD HANDLE
IT REQUIRES AN OLDER VERSION OF TORCHTEXT, HENCE THE TORCHTEXT.LEGACY IMPORTS
'''

class DataFrameDataset(data.Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
     
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]



class SeriesExample(data.Example):
    """Class to convert a pandas Series to an Example"""
  
    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex

