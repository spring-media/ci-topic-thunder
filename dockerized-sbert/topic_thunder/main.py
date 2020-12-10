from sentence_transformers import SentenceTransformer, models
import os
import tarfile, pickle
import io
import base64
import json
import re
from model_pipeline import NLPipe as ModelPipe
import time
import umap
from hdbscan import HDBSCAN
import pandas as pd
import pickle
import re, boto3
from sqlalchemy import create_engine
from os import environ
from dotenv import load_dotenv
from modules import utils
load_dotenv('./dockerized-sbert/.env.prod',verbose=True   )




S3_BUCKET = environ['S3_BUCKET'] if 'S3_BUCKET' in environ else 'fallback-test-value'
MODEL_PATH = environ['MODEL_PATH'] if 'MODEL_PATH' in environ else 'fallback-test-value'
HUD_HOST = environ['DB_HOST'] if 'DB_HOST' in environ else 'fallback-test-value'
HUD_DATABASE_NAME = environ['DB_NAME'] if "DB_NAME" in environ else 'fallback-test-value'
HUD_LOGIN = environ.get('DB_USER') if 'DB_USER' in environ else 'fallback-test-value'
HUD_PASSWORD = environ.get('DB_PASSWORD') if 'DB_PASSWORD' in environ else 'fallback-test-value'

SRC_TABLE_NAME = environ.get('ARTICLES_TABLE')
EMBEDDINGS_TABLE = environ.get('EMBEDDINGS_TABLE') # TODO: embeddings table name should be specified based on the model used for embedding generation
TOPIC_LABELS_TABLE = environ.get('TOPIC_LABELS_TABLE')

DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)
db_connection = create_engine(DB_URL)


class Manager:
    ''' Managment class for organizing flow of the operations.'''
    def __init__(self):
        # TODO: Read confif file from JSON adjusting encoder params
        # with open('./config.json', 'r') as f:
        #     self.config = json.load(f)
        #self.data= self._read_latest_articles()
        self.data = pd.DataFrame([])

    def init_pipeline(self):
        if not self.data.empty:
            self.pipeline=ModelPipe(self.data)


    def embedd_batch(self):
        columns=['embedding','article_created_at']
        if not self.data.empty:
            self.data['text'] =  self.data['seo_title'].apply(lambda x: utils.remove_seo_title_marker(x,True)) +". "+ self.data["text"]
        
            self.pipeline._store_headlines() 
            self.pipeline._store_created_at()
            embedded = self.pipeline.cleantext().embed(batch_size=50).normalize()
            
            return  pd.DataFrame(zip(embedded.value(),self.data.created_at.values),columns=columns,index=self.data.index)
        else:
            
            return pd.DataFrame([],columns=columns)

    @staticmethod
    def _save_embeddings_to_table(results_df):
        print(results_df)
        #res = pd.DataFrame(zip(results_df.value(),df.created_at.values),columns=['embedding','article_created_at'],index=df.index)
        res.embedding = res.embedding.apply(pickle.dumps)
        res.to_sql(name=EMBEDDINGS_TABLE, if_exists="append",con=db_connection)

    def _read_latest_articles(self,limit=250):
        """ Pull latest articles from the database. If limit is None then it pulls all articles.
            param: limit - tells how far in time shall we go back 
        """
        print(">>> Pulling latest articles from the database")
        SRC_SQL_QUERY = 'SELECT * FROM {} t '.format(SRC_TABLE_NAME)
        SRC_SQL_QUERY += 'ORDER BY t.created_at DESC'

        if limit:
            SRC_SQL_QUERY += ' LIMIT {}'.format(limit)

        TARGET_SQL_QUERY = "SELECT t.article_uid FROM {} t".format(EMBEDDINGS_TABLE)
        # TODO: Find a more efficient way to check if article is already embedded 
        df = pd.read_sql_query(SRC_SQL_QUERY, con=db_connection,parse_dates=["created_at"]).set_index("article_uid")
        target_df = pd.read_sql_query(TARGET_SQL_QUERY, con=db_connection, parse_dates=["article_created_at"]  )
        
        df.drop(target_df['article_uid'],inplace=True,errors="ignore")
        # TODO: Prep of befeore reading data. Casting? 
        print("Found {} new articles - {} are already in the db".format(df.shape[0],target_df.shape[0]))

        self.data = df
        self.init_pipeline()
        return df

    def _store_embeddings(self, df):
        try:
            df.embedding = df.embedding.apply(pickle.dump)
            df.to_sql(EMBEDDINGS_TABLE,if_exists="append",con=db_connection)
        except Exception as err:
            print("Could not store emebddings.")

    @staticmethod
    def _build_query_for_new_articles(article_ids, table_name):
        query = "SELECT * FROM {} t ".format(table_name)

        for (idx, article_id) in enumerate(article_ids):
            query += " WHERE  t.article_id = '{}' ".format(article_id)
            if idx == len(article_ids) - 1:
                query += " OR "
        SRC_SQL_QUERY += 'ORDER BY t.created_at DESC'

        query = query + ";"
        return query

    def _read_embeddings(self,article_ids=None):
        if article_ids is None:
            df = pd.read_sql_table(EMBEDDINGS_TABLE,con=db_connection,index_col=0,parse_dates=['article_created_at'])
        elif type(article_ids) == list:
            query = self._build_query_for_new_articles(article_ids,EMBEDDINGS_TABLE)
            df = pd.read_sql(query, con=db_connection,index_col=0,parse_dates=['article_created_at'])
        else:
            print("Wrong type of article_ids")
            raise Exception("Wrong type of article_ids")

        df.embedding = df.embedding.apply(pickle.loads)
        self.data = df
        if not df.empty:
            self.init_pipeline()

    def _store_topic_lables(self,article_ids,topic_labels):
        columns= ['article_uid', 'topic_label']
        df = pd.DataFrame(zip(article_ids,topic_labels),columns=columns).set_index('article_uid')
        df.to_sql(TOPIC_LABELS_TABLE,if_exists="append",con=db_connection)
    

if __name__ == "__main__":
    m = Manager()
    # df = m._read_latest_articles(2500)

    # res =m.embedd_batch()
    # if not res.empty:
    #      m._save_embeddings_to_table(res)
    m._read_embeddings()
    topic_labels = m.pipeline.cluster_hdbscan().value()
    m._store_topic_lables(m.data['article_uid'][:1000], topic_labels)