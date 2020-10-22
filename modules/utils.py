import nltk, pymysql,string,os,sys,re
import pandas as pd
from sqlalchemy import create_engine
import logging


import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from germalemma import GermaLemma
from nltk.util import ngrams
import numpy as np
import seaborn as sns
from collections import Counter
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from nltk.stem.snowball import GermanStemmer
stem = GermanStemmer()
import logging


from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer
from nltk.stem.cistem import Cistem
from datetime import datetime

def init_logger(name):
    '''set up training logger.'''
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    h = logging.StreamHandler(sys.stdout)
    h.flush = sys.stdout.flush
    logger.addHandler(h)
    return logger


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
    for news in df[col].values:
        news = news.lower()
        news = re.sub('[{}]'.format(re.escape("""▶︎►…!"#$%&'(!)*+,/:;<=>?@[\]^_`‚‘{|}~""")), '', news)
        news = re.sub("\d.","",news)
        news = re.sub("\n","",news)
        news = re.sub("                                     "," ",news)
        news =re.sub("bild.de","",news)
        news =re.sub("bildplus","",news)

        news =re.sub("\u2009"," ",news)
        news = re.sub("\s–\s"," ",news)
        news = re.sub("\d","",news)
        news = re.sub("\.\.\.",".",news)
        news = re.sub("(?:\s+)(-)","",news)#nextline moving

        news = re.sub("\s([A-Za-z])?\.","",news)

        news = re.sub('http\S+\s*', '', news)  # remove URLs
        news = news.replace('„',"")
        news = news.replace('‚',"").replace('\\‘',"")
        news = news.replace(u'\xa0', u' ')

        news = news.replace('“',"")
        news = news.replace('bild.de',"")
        news = news.replace('\u2005',"")
        news=re.sub("  "," ",news)
        news = news.rstrip().lstrip()
        if stemming:
            news = " ".join([stem.stem(word) for word in news.split()])
            
        corpus.append(news)
    return corpus


def remove_seo_title_marker(headline,remove_section=False):
    tmp = headline.replace('bild.de',"")
    tmp = tmp.replace('Bild.de',"")
    tmp = tmp.replace('*** BILDplus Inhalt ***',"")
    tmp = tmp.replace(u'\xa0', u' ')
    if remove_section:
        tmp=[wrds.lstrip().rstrip() for wrds in tmp.split("-")[:-2]]
    else:
        tmp=[wrds.lstrip().rstrip() for wrds in tmp.split("-")[:-1]]
    return " ".join(tmp).lstrip().rstrip()



def load_text_data(path="../data/bild_articles.csv"):
    try:
        df=pd.read_csv(path,index_col=0)
        df.created_at =pd.to_datetime(df.created_at,dayfirst=True)
        return df
    except FileNotFoundError as err:
        print("Could not find path")
    
def load_raw_data(path):
    df=pd.read_csv(path,index_col=0)
    df.created_at =pd.to_datetime(df.created_at,dayfirst=True)
    return df

def load_labeled_data(path="../data/labeled_test_clusters.csv"):
    return load_raw_data(path)


def get_sentences_from_text(text):
    return [s.rstrip().lstrip() for s in text.split(".") if s]


def parse_google_named_entities(json_obj):
    #Parse objets and deduplicate list of them 
    parse_named_entity = lambda ne: {"text":ne["name"],"type":ne["type"]}
    list_of_objects = [parse_named_entity(ne) for ne in json.loads(json_obj)[0]["entities"] if "bild" not in ne["name"].lower() ]
    return [i for n, i in enumerate(list_of_objects) if i not in list_of_objects[n + 1:]]

        
# Prepare data
def link_to_raw_data(data_to_viz,df,cluster_labels):

    result = pd.DataFrame(data_to_viz, columns=['x', 'y','z'])
    result['labels'] = cluster_labels
    result['headline'] = df["seo_title"].values
    result['seo_title'] = df["headline"].values
    result['text'] = df["text"].values
    result['id'] = df.index.values

    result['created_at'] = df["created_at"].dt.date.values
    result.sort_values(by="created_at")
    result['created_at'] = result.created_at.apply(str)
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    print("Outliers: {} | Clustered: {} | {} \n Cluster count: {} ".format(len(outliers),len(clustered),
                                                                     (len(clustered)/(len(outliers)+len(clustered)))
                                                                     ,len(clustered.labels.unique())))

    return result



def c_tf_idf(documents, m, ngram_range=(1, 1),remove_stop_words=True):
    if remove_stop_words:
        def remove_stop_words(doc):
            for sword in STOPWORDS:
                doc=doc.replace(sword,"")
                return doc
        documents=np.array(list(map(remove_stop_words, documents)))
    
    count = CountVectorizer(ngram_range=ngram_range).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes