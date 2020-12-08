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
load_dotenv(verbose=True)




S3_BUCKET = environ['S3_BUCKET'] if 'S3_BUCKET' in environ else 'fallback-test-value'
MODEL_PATH = environ['MODEL_PATH'] if 'MODEL_PATH' in environ else 'fallback-test-value'
HUD_HOST = environ['DB_HOST'] if 'DB_HOST' in environ else 'fallback-test-value'
HUD_DATABASE_NAME = environ['DB_NAME'] if "DB_NAME" in environ else 'fallback-test-value'
HUD_LOGIN = environ.get('DB_USER') if 'DB_USER' in environ else 'fallback-test-value'
HUD_PASSWORD = environ.get('DB_PASSWORD') if 'DB_PASSWORD' in environ else 'fallback-test-value'

SRC_TABLE_NAME = environ.get('ARTICLES_TABLE')
EMBEDDINGS_TABLE = environ.get('EMBEDDINGS_TABLE')

DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)
db_connection = create_engine(DB_URL)


class Manager:
    def __init__(self):
        pass
        # with open('./config.json', 'r') as f:
        #     self.config = json.load(f)

    def embedd_batch(self,df):

        df['text'] =  df['seo_title'].apply(lambda x: utils.remove_seo_title_marker(x,True)) +". "+ df["text"]
    
        nlpipe = ModelPipe(df)
        nlpipe._store_headlines() 
        nlpipe._store_created_at()

        embedded = nlpipe.cleantext().embed(batch_size=50).normalize()
        self.pipeline = embedded
        res = pd.DataFrame(zip(embedded.value(),df.created_at.values),columns=['embedding','article_created_at'],index=df.index)
        return res

    @staticmethod
    def _save_embeddings_to_table(results_df):
        print(results_df)
        #res = pd.DataFrame(zip(results_df.value(),df.created_at.values),columns=['embedding','article_created_at'],index=df.index)
        res.embedding = res.embedding.apply(pickle.dumps)
        res.to_sql(name=EMBEDDINGS_TABLE, if_exists="append",con=db_connection)

    @staticmethod
    def _build_query_for_new_articles(article_ids, table_name):
        query = "SELECT * FROM {} arts ".format(table_name)

        for (idx, article_id) in enumerate(article_ids):
            query += " WHERE  arts.article_id = 'article_id' "
            if idx == len(article_ids) - 1:
                query += " OR "

        query = query + ";"
        return query

    def _pull_latest_articles(self,limit=250):
        """ Pull latest articles from the database. If limit is None then it pulls all articles."""
        print(">>> Pulling latest articles from the database")
        SRC_SQL_QUERY = 'SELECT * FROM {}'.format(SRC_TABLE_NAME)
        if limit:
            SRC_SQL_QUERY += ' LIMIT {}'.format(limit)

        TARGET_SQL_QUERY = "SELECT 'article_uid' FROM {}".format(EMBEDDINGS_TABLE)

        df = pd.read_sql_query(SRC_SQL_QUERY, con=db_connection,parse_dates=["created_at"]).set_index('article_uid')
        target_df = pd.read_sql_query(TARGET_SQL_QUERY, con=db_connection, parse_dates=["article_created_at"])

        #print(target_df["article_uid"].values)
        #print(df.drop(target_df['article_uid']))
        # TODO: Prep of befeore reading data. Castinnng 
        print("Found {} new articles - {} are already in the db".format(df.shape[0],target_df.shape[0]))
        return df

    def _store_embeddings(self, df):
        try:
            df.embedding = res.embedding.apply(pickle.dump)
            df.to_sql(EMBEDDINGS_TABLE,if_exists="append",con=db_connection)
        except Exception as err:
            print("Could not store emebddings.")


if __name__ == "__main__":
    m = Manager()
    df = m._pull_latest_articles()
    res =m.embedd_batch(df)
    
    m._save_embeddings_to_table(res)
    m.pipeline.cluster().value()