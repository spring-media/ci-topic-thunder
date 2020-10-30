# ci-topic-thunder

![logo](./topic-thunder-logo.jpg)
Repo for storing code related to topic thunder master thesis project.


### Auxiliary 
- MLFlow server : http://ec2-18-184-134-25.eu-central-1.compute.amazonaws.com
    - Credentials: mlflow / mlflow
- S3 Buckets: 
    - s3://ci-topic-thunder/ - stores models and data files (to big for git)
    - s3://ci-mlflow-server/ - stores MLFlow artifacts

### Dir structure
- code - contains all the Python modules and classes needed to run exeperiments.
- data -  CSV files needed to run experiments as well as scripts to download them.
- models - models and scripts needed to download them
- papers - research papers and PDF resources for thesis
- notebooks - notebooks with experiments


```

# Install requirements
$ pip install -r requirements.txt

# Make sure you have set up the AWS credentials in ~/.aws/credentials file. If not install AWS CLI and run:
$ aws configure

# Synchronising data and models folder with the latest version of data from S3:
$ ./sync.sh
``` 



## Resources:
#### Articles:
- SentenceBERT TLDR - https://medium.com/dair-ai/tl-dr-sentencebert-8dec326daf4e 
- BERTopics - https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
- Comprehesive overview of documents embeding techniques with lots of references and comparisons:  https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d
- https://supernlp.github.io/2018/11/26/sentreps/
- http://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/
- Explanation of the SBERT: https://medium.com/genei-technology/richer-sentence-embeddings-using-sentence-bert-part-i-ce1d9e0b1343
- MLFLOW Setup :https://github.com/ymym3412/mlflow-docker-compose#3-Set-up-NGINX-Basic-Authentication


- Boosting the quality of embedding by simple prepossessing : https://github.com/vyraun/Half-Size

- Plotly and dash for NLP viz: https://medium.com/plotly/nlp-visualisations-for-clear-immediate-insights-into-text-data-and-outputs-9ebfab168d5b

- FAISS from Facebook AI - A library for efficient similarity search and clustering of dense vectors. - https://github.com/facebookresearch/faiss

#### Papers:
- PV for Vox Media + Wikipedia: 
- https://www.catalyzex.com/paper/arxiv:1208.4411
- https://www.sciencedirect.com/science/article/pii/S1532046416300442
- https://reader.elsevier.com/reader/sd/pii/S1532046416300442
- Effective Dimensionality reduction for word embeddings - https://www.aclweb.org/anthology/W19-4328/
- Text Summarization with pretrained encoders (can be used fror cluster description) - https://arxiv.org/abs/1908.08345 

