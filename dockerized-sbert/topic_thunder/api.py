import collections, json, logging, os, re, string, sys

sys.path.append("../")
sys.path.append(os.path.dirname(__file__))

from typing import Dict
from fastapi import Depends, FastAPI
import uvicorn,json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from model_pipeline import NLPipe
from main import Manager
load_dotenv()

app = FastAPI(debug=True)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/start')
async def start():
    mgr = Manager()
    results = mgr._pull_latest_articles()
    #results = NLPipe()
    #emebddings = SBERT.encode(DL.data)
    #results = dict(zip(DL.index, emebddings))
    #print(results)
    return json.dump(results)


if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0' ,debug=True,port=8080)
