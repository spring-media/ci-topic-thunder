import time
import umap
from hdbscan import HDBSCAN
import pandas as pd
import pickle
import re, boto3
from sqlalchemy import create_engine
from os import environ

S3_BUCKET = environ['S3_BUCKET'] if 'S3_BUCKET' in environ else 'fallback-test-value'
MODEL_PATH = environ['MODEL_PATH'] if 'MODEL_PATH' in environ else 'fallback-test-value'
HUD_HOST = environ['DB_HOST'] if 'DB_HOST' in environ else 'fallback-test-value'
HUD_DATABASE_NAME = environ['DB_NAME'] if "DB_NAME" in environ else 'fallback-test-value'
HUD_LOGIN = environ.get('DB_USER') if 'DB_USER' in environ else 'fallback-test-value'
HUD_PASSWORD = environ.get('DB_PASSWORD') if 'DB_PASSWORD' in environ else 'fallback-test-value'

SRC_TABLE_NAME = environ.get('ARTICLES_TABLE')
EMBEDDINGS_TABLE = environ.get('EMBEDDINGS_TABLE')

DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)
db_connection = create_engine(DB_URL, echo=True)
s3 = boto3.client('s3')

s3.list_buckets()

class DataLoader:
    def __init__(self):

        df = self._pull_latest_articles()

        df['_input'] = df['seo_title'].apply(lambda x: self._remove_seo_title_marker(x, True)) + ". " + df["text"]
    
        self.index = df.index.to_list()
        self.data = self.preprocess_articles_for_bert(df, col="_input")

        for x in range(5):
            print("{}. {}...".format(x, self.data[x][:250]))
    
    @staticmethod
    def _remove_seo_title_marker(headline, remove_section=False):
        """
        This function removes the suffix category tag from the seo_title.
        :param headline:
        :param remove_section:
        :return:
        """
        tmp = headline.replace('bild.de', "")
        tmp = tmp.replace('Bild.de', "")
        tmp = tmp.replace('*** BILDplus Inhalt ***', "")
        tmp = tmp.replace(u'\xa0', u' ')
        if remove_section:
            tmp = [wrds.lstrip().rstrip() for wrds in tmp.split("-")[:-2]]
        else:
            tmp = [wrds.lstrip().rstrip() for wrds in tmp.split("-")[:-1]]
        return " ".join(tmp).lstrip().rstrip()
    
    @staticmethod
    def _build_query_for_new_articles(article_ids, table_name):
        query = "SELECT * FROM {} arts ".format(table_name)

        for (idx, article_id) in enumerate(article_ids):
            query += " WHERE  arts.article_id = 'article_id' "
            if idx == len(article_ids) - 1:
                query += " OR "

        query = query + ";"
        return query

    def _pull_latest_articles(self):
        print(">>> Prep started")
        SRC_SQL_QUERY = 'SELECT * FROM {} limit 250'.format(SRC_TABLE_NAME)
        TARGET_SQL_QUERY = "SELECT 'article_uid' FROM {}".format(EMBEDDINGS_TABLE)

        df = pd.read_sql_query(SRC_SQL_QUERY, con=db_connection,parse_dates=["created_at"]).set_index('article_uid')
        target_df = pd.read_sql_query(TARGET_SQL_QUERY, con=db_connection,parse_dates=["article_created_at"])

        print(target_df["article_uid"].values)
        print(df.drop(target_df['article_uid']))
        # TODO: Prep of befeore reading data
        return df

    def _store_embeddings(self, df):
        
        try:
            df.to_sql(EMBEDDINGS_TABLE,if_exists="append",con=db_connection)
        except SQLAlchemyError as err:
            print("Could not store emebddings.")


    def preprocess_articles_for_bert(self, articles, col="text", lower=False):
        """
        Main text preprocessing function used to preprare article text for the SentenceBert.
        It does not process the article in a hard way. Instead, it removes double spaces, wierd characters, unencoded HTML tags, trilple dots.
            Football scores like (2:2) and 3:0-Sieg are removed
            Bild and bild plus references are removed.
            Double words i.e Potsdam/Brandenburg are split
            Bild references are removed

        :param articles: DataFrame with articles
        :param col: Which column contains the article text
        :param lower: Should we lower at the end?
        :return:
        """
        corpus = []

        for news in articles[col].values:
            news = re.sub(r" \(\d+\)", "", news)  # Numbers in brackets ie ages
            news = re.sub(r"\d+:+\d+-", "", news)  # Remove scores from i.e 2:3-Sieg
            news = re.sub(r"\(+\d+:+\d\)+", "", news)  # Scores in brackets

            # Drop Weird characters
            news = re.sub('[{}]'.format(re.escape(r'–•™©®●▶︎►…"$%&()*œ†¥+:;µ≤≥æ«‘π<=>[\]^_`‚‘{|}~\'')), '', news)

            news = re.sub("bild.de", "", news, flags=re.IGNORECASE)
            news = re.sub("bildplus", "", news, flags=re.IGNORECASE)
            news = re.sub("bild plus", "", news, flags=re.IGNORECASE)
            # news = re.sub(r"ß", "ss", news)  # Remove ß

            # news = re.sub(r"\?", ".", news) # ?
            news = re.sub(r"\!", ".", news)  # !

            news = re.sub(r"\n", "", news)  # Remove newlines

            for x in range(3, 20, 1):
                news = re.sub(" " * x, " ", news)  # Remove 3,4,5...20 consecutive spaces

            news = re.sub(r"\.\.\.", ".", news)  # Replace three dots
            news = re.sub(r"(?:\s+)(-)", "", news)  # nextline moving
            news = re.sub(r'http\S+\s*', '', news)  # remove URLs
            news = re.sub(r'^\$[0-9]+(\.[0-9][0-9])?$', '', news)  # remove dollar prefixed numbers
            news = re.sub(r'^[0-9]+(\.[0-9][0-9])?\€', '', news)  # remove euro prefixed numbers

            news = news.replace(u' €', u' EURO')
            news = news.replace(u' £', u' POUNDS')
            news = news.replace(u' $', u' POUNDS')

            news = news.replace('“', "")
            news = news.replace('„', "")

            news = news.replace('\xa0', '')
            news = news.replace('\xad', '')
            news = news.replace('\u00ad', '')
            news = news.replace('\u2005', "")  # Remove some unicode characters
            news = re.sub("\u2009", " ", news)  # remove weird unicode chars

            news = news.replace(' . ', ". ")  # Unnecessary spaces
            news = news.replace('. ', ". ")  # Unnecessary spaces
            news = re.sub(r'(\w+)(\/)(\w+)', r"\g<1> \g<3>", news)  # Double words i.e Potsdam/Brandenburg are split

            news = re.sub(" {2}", " ", news)  # Remove doube spaces
            news = re.sub(r"\.\.", ".", news)
            news = re.sub(r" \. {2}", ". ", news)

            news = news.rstrip().lstrip()

            if lower:
                news = news.lower()
            corpus.append(news)
        return corpus


