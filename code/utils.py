import fasttext.util
import nltk, pymysql,string
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from germalemma import GermaLemma
from nltk.util import ngrams
import numpy as np
import seaborn as sns
from collections import Counter
from hdbscan import HDBSCAN
from gensim.models.wrappers import FastText
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from nltk.stem.snowball import GermanStemmer
stem = GermanStemmer()

from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer
from nltk.stem.cistem import Cistem
import os,sys,re,string

sys.path.append(os.path.dirname(__file__))
with open(os.path.dirname(__file__)+"/german_stopwords_plain.txt") as f:
    STOPWORDS = [line.strip() for line in f if not line.startswith(";")]
    STOPWORDS += ["dass", "", "/t", "   ", "...", "worden", "jahren", "jahre", "jahr",
                  "heißt", "heißen", "müsse", "prozent", "BILD", "etwas"]
    STOPWORDS = set(STOPWORDS)

print("Number of stopwords {}".format(len(STOPWORDS)))


tokenizer = TreebankWordTokenizer()  # RegexpTokenizer(r'\w+')


def preprocess_and_tokenize_text(df,col="text"):
    corpus = []
    punctuation = string.punctuation
    punctuation.remove(".")

    for news in df[col]:
        words = []
        news = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', news)
        news = [''.join(c for c in s if c not in punctuation) for s in news]
        news = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", news)
        news = re.sub('http\S+\s*', '', news)  # remove URLs
        news = re.sub(r'\b\d+\b', '', news)

        for w in tokenizer.tokenize(news):
            if (w not in STOPWORDS) and (not w.isdigit()) and (w.isalpha()) and len(w) > 1:
                w = stem.stem(w)
                words.append(w)
        corpus.append(words)
    return corpus



def preprocess_text(df,col="text",stemming = False):
    corpus = []
    punctuation = string.punctuation
    punctuation.replace(".","")
    for news in df[col]:
        news = news.lower()
        news = re.sub('[{}]'.format(re.escape("""▶︎►…!"#$%&'(!)*+,/:;<=>?@[\]^_`‚‘{|}~""")), '', news)
        news = re.sub("\d.","",news)
        news = re.sub("\n","",news)
        news = re.sub("                                     "," ",news)
        news =re.sub("bild.de","",news)
        news =re.sub("\u2009"," ",news)
        news = re.sub("\s–\s"," ",news)
        news = re.sub("\d","",news)
        news = re.sub("\.\.\.",".",news)
        news = re.sub("(?:\s+)(-)","",news)#nextline moving

        news = re.sub("\s([A-Za-z])?\.","",news)

        news = re.sub('http\S+\s*', '', news)  # remove URLs
        #news= news.replace('..','.').replace(' . ','.').replace("   "," ")
        #news = news.replace("▶︎","")
        news = news.replace('„',"")
        news = news.replace('‚',"").replace('\\‘',"")

        news = news.replace('“',"")
        news = news.replace('bild.de',"")
        news = news.replace('\u2005',"")
        news=re.sub("  "," ",news)
        news = news.rstrip().lstrip()
        if stemming:
            news = " ".join([stem.stem(word) for word in news.split()])
            
        corpus.append(news)
    return corpus



def load_text_data():
    return pd.read_csv("./bild_articles.csv",index_col=0)

def get_sentences_from_text(text):
    return [s.rstrip().lstrip() for s in text.split(".") if s]

