# ci-topic-thunder

![logo](./topic-thunder-logo.jpg)
Repo for storing code related to topic thunder master thesis project.
Notes from the user inteviews:
- https://moveoffice-my.sharepoint.com/:w:/g/personal/tomasz_tkaczyk_axelspringer_com/ESnj_G2zvulIsTa-nxgVV5YBNqeWScKDwGgmvIy0Z7e4MA?e=Dx9tGW


## Prototype
![Prototype](./Topic_thunder.png)

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

### Data files:
This directory is syncronised with S3 bucket folder. It contians all the raw data files.

- dump_last-160k_31-10.csv	- most up-to-date data export from 31.10.2020. It Contains a joined table of articles + named entities for every article in the database. 
- bild_articles.csv - older data dump from 9th of September. It has ca. 8k articles less the newes one from the dump. 
- embeddings_bild_articles.npy -	embeddings of all the articles form bild_articles.csv
- headlines-2016-deepl.csv - set of ca. 750 headline pairs from the SentEval dataset. Translated using DeepL. [Original File](https://github.com/brmson/dataset-sts/blob/master/data/sts/semeval-sts/2016/headlines.test.tsv)
- labeled_test_clusters.csv	- contains 240 labeled articles (20 Clusters). Used for parameter search and evaluation of the clustering performance
- raw_article_entities.csv - named entities matiching the labeled dataset above.
- embeddings.npy - numpu dump of all emebeddings generated for all articles in bild_articles.csv. (Input size was 128 characters )

### Models
This directory is syncronised with S3 bucket folder. It contians all the models and artifacts files.


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
- Papers with code: Sentence Emebddings - https://paperswithcode.com/task/sentence-embeddings

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

