import collections, json, logging, os, re, string, sys

sys.path.append("../")
sys.path.append(os.path.dirname(__file__))
from sentence_transformers import SentenceTransformer,util

from fastapi import FastAPI
from modules import utils,modeling
import pandas as pd
import numpy as np
import sentence_transformers,umap,pickle

app = FastAPI()
df= utils.load_text_data("../data/Result_31.csv").head(50000).sort_values("created_at",ascending=False).drop_duplicates("headline").dropna()

_pdf = pd.DataFrame(np.load("../models/cross-en-de-roberta-sentence-transformer/Top-50k_articles_embddings_ids.npy",allow_pickle=True),columns=["article_id","embedding"]).set_index("article_id")
#_pdf=pd.merge(_pdf,df,on="article_id")


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/search")
async def search(query_article_id):
    return json.dumps({"res":"sds"})

@app.get("/search/{query_article_id}")
async def read_item(query_article_id):
    res = query_embeddings(query_article_id,_pdf,df)
    return res.to_dict('records')



def query_embeddings(query_article_id,corpus,raw_df,top_k=10):
    search_results = util.semantic_search(np.array(_pdf.loc[query_article_id].embedding,dtype="double"),
                         np.stack(corpus.embedding.values).astype(np.double),top_k=top_k)
    query_res_df = pd.DataFrame([(_pdf.iloc[res['corpus_id']].name,res['score']) for res in search_results[0]],columns=['article_uid','score']).set_index("article_uid")
    return query_res_df.join(raw_df)[["seo_title",'headline',"created_at","score"]].sort_values(by=['score',"created_at"],ascending=False)

#NIZZA =np.array(_pdf.loc["65f65b24d9b4530204775aa8d56eb3a0ad8e1be20e8f756c003684fb512a44ad"].embedding,dtype="double")
#search_results = utils.semantic_search(NIZZA,
 #                        np.stack(_pdf.embedding.values).astype(np.double),top_k=12)

#query_res_df = pd.DataFrame([(_pdf.iloc[res['corpus_id']].name,res['score']) for res in search_results[0]],columns=['article_uid','score']).set_index("article_uid")
#query_res_df.join(df)[["seo_title",'headline',"created_at","score"]].sort_values(by=['score',"created_at"],ascending=False)

