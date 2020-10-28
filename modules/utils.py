import collections
import os, re, string, sys, logging, json, sklearn

from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score, silhouette_score

sys.path.append("../")
sys.path.append(os.path.dirname(__file__))
from modules import modeling
import numpy as np
import pandas as pd
from nltk.stem.snowball import GermanStemmer
from sklearn.feature_extraction.text import CountVectorizer

stem = GermanStemmer()

from nltk.tokenize import TreebankWordTokenizer


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


with open(os.path.dirname(__file__) + "/german_stopwords_plain.txt") as f:
    STOPWORDS = [line.strip() for line in f if not line.startswith(";")]
    STOPWORDS += ["dass", "", "/t", "   ", "...", "worden", "jahren", "jahre", "jahr",
                  "heißt", "heißen", "müsse", "prozent", "BILD", "etwas"]
    STOPWORDS = set(STOPWORDS)

print("Number of stopwords {}".format(len(STOPWORDS)))

tokenizer = TreebankWordTokenizer()  # RegexpTokenizer(r'\w+')


def preprocess_and_tokenize_text(df, col="text"):
    corpus = []
    punctuation = string.punctuation
    punctuation.remove(".")

    for news in df[col]:
        words = []
        news = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', news)
        news = [''.join(c for c in s if c not in punctuation) for s in news]
        news = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", news)
        news = re.sub(r'http\S+\s*', '', news)  # remove URLs
        news = re.sub(r'\b\d+\b', '', news)

        for w in tokenizer.tokenize(news):
            if (w not in STOPWORDS) and (not w.isdigit()) and (w.isalpha()) and len(w) > 1:
                w = stem.stem(w)
                words.append(w)
        corpus.append(words)
    return corpus


def preprocess_articles_for_bert(articles, col="text", lower=False):
    corpus = []

    for news in articles[col].values:
        news = re.sub(r" \(\d+\)", "", news)  # Numbers in brackets ie ages
        news = re.sub(r"\d+:+\d+-", "", news) # Remove scores from i.e 2:3-Sieg
        news = re.sub(r"\(+\d+:+\d\)+", "", news)  # Scores in brackets

        news = re.sub('[{}]'.format(re.escape(r'–„▶︎►…"$%&()*+:;<=>[\]^_`‚‘{|}~\'')), '', news)
        news = re.sub("bild.de", "", news, flags=re.IGNORECASE)
        news = re.sub("bildplus", "", news, flags=re.IGNORECASE)


        #news = re.sub(r"\?", ".", news) # ?
        #news = re.sub(r"!", ".", news) # !


        news = re.sub(r"\n", "", news)
        news = re.sub("                                     ", " ", news)
        news = re.sub(r"\.\.\.", ".", news)
        news = re.sub(r"(?:\s+)(-)", "", news)  # nextline moving
        news = re.sub(r'http\S+\s*', '', news)  # remove URLs
        news = news.replace(u'\xa0', u' ')

        # news = news.replace('“', "")
        # news = news.replace('„', "")

        news = news.replace('\u2005', "")
        news = re.sub("\u2009", " ", news) # remove weird unicode chars
        news = news.replace('', "")
        # news = news.replace('‚', "").replace('\\‘', "")
        news = news.replace(' . ', ". ")
        news = news.replace('. ', ". ")
        news = re.sub(r'(\w+)(\/)(\w+)', "\g<1> \g<3>", news)

        news = re.sub("  ", " ", news)  # Remove doube spaces
        news = re.sub(r"\.\.", ".", news)
        news = re.sub(r" \.  ", ". ", news)

        news = news.rstrip().lstrip()
        if lower:
            news = news.lower()
        corpus.append(news)
    return corpus


def preprocess_text(df, col="text", stemming=False):
    corpus = []
    punctuation = string.punctuation
    punctuation.replace(".", "")  # Remove Dot from the punctuation list so it wont be removed from the articles.
    for news in df[col].values:
        news = news.lower()
        news = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', news)
        news = re.sub(r"\d.", "", news)
        news = re.sub(r"\n", "", news)
        news = re.sub("                                     ", " ", news)
        news = re.sub("bild.de", "", news)
        news = re.sub("bildplus", "", news)

        news = re.sub("\u2009", " ", news)
        news = re.sub(r"\s–\s", " ", news)
        news = re.sub(r"\d", "", news)
        news = re.sub(r"\.\.\.", ".", news)
        news = re.sub(r"(?:\s+)(-)", "", news)  # nextline moving

        news = re.sub(r"\s([A-Za-z])?\.", "", news)

        news = re.sub(r'http\S+\s*', '', news)  # remove URLs
        news = news.replace('„', "")
        news = news.replace('‚', "").replace('\\‘', "")
        news = news.replace(u'\xa0', u' ')

        news = news.replace('“', "")
        news = news.replace('\u2005', "")
        news = re.sub("  ", " ", news)
        news = news.rstrip().lstrip()
        if stemming:
            news = " ".join([stem.stem(word) for word in news.split()])

        corpus.append(news)
    return corpus


def remove_seo_title_marker(headline, remove_section=False):
    tmp = headline.replace('bild.de', "")
    tmp = tmp.replace('Bild.de', "")
    tmp = tmp.replace('*** BILDplus Inhalt ***', "")
    tmp = tmp.replace(u'\xa0', u' ')
    if remove_section:
        tmp = [wrds.lstrip().rstrip() for wrds in tmp.split("-")[:-2]]
    else:
        tmp = [wrds.lstrip().rstrip() for wrds in tmp.split("-")[:-1]]
    return " ".join(tmp).lstrip().rstrip()


def load_text_data(path="../data/bild_articles.csv"):
    try:
        df = pd.read_csv(path, index_col=0)
        df.created_at = pd.to_datetime(df.created_at, dayfirst=True)
        return df
    except FileNotFoundError as err:
        print("Could not find {}".format(path))


def load_raw_data(path):
    df = pd.read_csv(path, index_col=0)
    df.created_at = pd.to_datetime(df.created_at, dayfirst=True)
    return df


def load_labeled_data(path="../data/labeled_test_clusters.csv"):
    return load_raw_data(path)


def get_sentences_from_text(text):
    return [s.rstrip().lstrip() for s in text.split(".") if s]


def parse_google_named_entities(json_obj):
    # Parse objets and deduplicate list of them
    parse_named_entity = lambda ne: {"text": ne["name"], "type": ne["type"]}
    list_of_objects = [parse_named_entity(ne) for ne in json.loads(json_obj)[0]["entities"] if
                       "bild" not in ne["name"].lower()]
    return [i for n, i in enumerate(list_of_objects) if i not in list_of_objects[n + 1:]]


# Prepare data
def link_to_raw_data(data_to_viz, df, cluster_labels):
    if data_to_viz.shape[1] == 3:
        result = pd.DataFrame(data_to_viz, columns=['x', 'y', 'z'])
    else:
        result = pd.DataFrame(data_to_viz, columns=['x', 'y'])
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
    print("Outliers: {} | Clustered: {} | {} \n Cluster count: {} ".format(len(outliers), len(clustered),
                                                                           (len(clustered) / (
                                                                                   len(outliers) + len(clustered)))
                                                                           , len(clustered.labels.unique())))

    return result


# Prepare data
def relink_data_after_clustering(data_to_viz, df, cluster_labels):
    if data_to_viz.shape[1] == 3:
        result = pd.DataFrame(data_to_viz, columns=['x', 'y', 'z'])
    else:
        result = pd.DataFrame(data_to_viz, columns=['x', 'y'])
    result['topic_number'] = cluster_labels
    result['headline'] = df["headline"].values
    result['seo_title'] = df["seo_title"].values
    result['raw_text'] = df["text"].values
    result['article_uid'] = df.index.values
    result["kicker_headline_ne"] = df.kicker_headline_NER.values
    result["text_ne"] = df.text_NER.values
    result["seo_title_ne"] = df.seo_title_NER.values

    result['created_at'] = df["created_at"].dt.date.values

    result.sort_values(by="created_at")
    result['created_at'] = result.created_at.apply(str)
    outliers = result.loc[result.topic_number == -1, :]
    clustered = result.loc[result.topic_number != -1, :]
    print("Outliers: {} | Clustered: {} | {} \n Cluster count: {} ".format(len(outliers), len(clustered),
                                                                           (len(clustered) / (
                                                                                   len(outliers) + len(clustered)))
                                                                           , len(clustered.topic_number.unique())))

    return result


flatten = lambda t: [item for sublist in t for item in sublist]  # Flatten list of lists
dedupe = lambda list_of_objects: [i for n, i in enumerate(list_of_objects) if
                                  i not in list_of_objects[n + 1:]]  # Deduplicated list of objects


def parse_google_named_entities(json_obj, deduplicate=False):
    # Parse objets and deduplicate list of them
    parse_named_entity = lambda ne: {"text": ne["name"], "type": ne["type"]}
    list_of_objects = [parse_named_entity(ne) for ne in json.loads(json_obj)[0]["entities"] if
                       "bild" not in ne["name"].lower()]
    if not deduplicate:
        return dedupe(list_of_objects)
    else:
        return list_of_objects


def c_tf_idf(documents, m, ngram_range=(1, 1), remove_stop_words=True):
    if remove_stop_words:
        def remove_stop_words(doc):
            for sword in STOPWORDS:
                doc = doc.replace(sword, "")
                return doc

        documents = np.array(list(map(remove_stop_words, documents)))

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
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df, col="Topic"):
    topic_sizes = (df.groupby([col])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes


def mlflow_run_model_eval(mlflow, embeddings, df, max_pooling, min_cluster_size, N_COMPONENTS, alpha, min_samples,
                          n_neighbors):
    with mlflow.start_run():
        # ctr=0
        mlflow.log_param(key="max_pooling", value=max_pooling)
        mlflow.log_param(key="min_cluster_size", value=min_cluster_size)
        mlflow.log_param(key="N_COMPONENTS", value=N_COMPONENTS)
        mlflow.log_param(key="alpha", value=alpha)
        mlflow.log_param(key="min_samples", value=min_samples)

        mlflow.log_param(key="n_neighbors", value=n_neighbors)

        results, cluster_labels = modeling.cluster_and_reduce(embeddings, n_components_clustering=N_COMPONENTS,
                                                              min_cluster_size=min_cluster_size,
                                                              n_neighbors=n_neighbors,
                                                              min_samples=min_samples, alpha=alpha)
        mlflow.log_metric(key="completeness_score", value=completeness_score(cluster_labels, y.values))
        mlflow.log_metric(key="v_measure_score", value=v_measure_score(cluster_labels, y.values))
        mlflow.log_metric(key="homogeneity_score", value=homogeneity_score(cluster_labels, y.values))
        mlflow.log_metric(key="normalized_mutual_info_score", value=v_measure_score(cluster_labels, y.values))

        summarized_clusters = dict(collections.Counter(cluster_labels))
        try:
            mlflow.log_metric(key="outliers_ratio", value=(summarized_clusters[-1] / len(cluster_labels)))
        except ZeroDivisionError as err:
            mlflow.log_metric(key="outliers_ratio", value=0)

        mlflow.log_metric(key="unique_clusters", value=len(summarized_clusters.items()))
        # mlflow.log_metric(key="cluters_ratio_to_GT",
        #                   value=(len(summarized_clusters.items()) - 1) / len(y_summarized_clusters.items()))
        mlflow.log_metric(key="silhuette_score", value=silhouette_score(_X, cluster_labels))

        results = link_to_raw_data(results, df, cluster_labels)

        modeling.scatter_plot(results, save_fig=True)
        mlflow.log_artifact("./tmp_scatter_plot.html")
