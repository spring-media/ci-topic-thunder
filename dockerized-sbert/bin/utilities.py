import pickle, boto3, sqlalchemy,fire,os,dotenv
from os import environ
import pandas as pd
import numpy as np

dotenv.load_dotenv()
HUD_HOST = environ['DB_HOST'] if 'DB_HOST' in environ else 'fallback-test-value'
HUD_DATABASE_NAME = environ['DB_NAME'] if "DB_NAME" in environ else 'fallback-test-value'
HUD_LOGIN = environ.get('DB_USER') if 'DB_USER' in environ else 'fallback-test-value'
HUD_PASSWORD = environ.get('DB_PASSWORD') if 'DB_PASSWORD' in environ else 'fallback-test-value'

SRC_TABLE_NAME = environ.get('ARTICLES_TABLE')
EMBEDDINGS_TABLE = environ.get('EMBEDDINGS_TABLE')

DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)


def load_precomputed_embeddings(path_to_npy_file):
    """ Load the precomputed embeddings from a numpy file to the database """ 
    if path_to_npy_file and os.path.isfile(path_to_npy_file):
        try:
            embeddings = np.load(path_to_npy_file,allow_pickle=True)
            columns = ['article_uid','embedding','article_created_at'][:(embeddings.shape[1])]
            df = pd.DataFrame(embeddings,columns=columns)
            df.embedding = df.embedding.apply(pickle.dumps)
            df = df.drop_duplicates(subset=['article_uid'])
            db_connection = sqlalchemy.create_engine(DB_URL)
            df.to_sql(EMBEDDINGS_TABLE,if_exists="append",con=db_connection,index=False)
        except Exception as err:
            print("Could not store emebddings.",err)
    else:
        print(" File not found.")

if __name__ == "__main__":
    fire.Fire(load_precomputed_embeddings)
    load_precomputed_embeddings('./dockerized-sbert/bin/dump-old-200k_articles_embddings_ids.npy')