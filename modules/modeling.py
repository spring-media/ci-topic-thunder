import time
import umap

import numpy as np
import plotly.express as px
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from modules import utils
import pickle
# tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")


def get_sentence_embeddings(array, sbert_worde_embedding_model, pooling_mode_max_tokens=False):
    """
        This function takes array of (preprocessed) sentence embeddings, a model and one paramter and returns the embeddings
    :param array:
    :param sbert_worde_embedding_model:
    :param pooling_mode_max_tokens:
    :return:
    """
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(sbert_worde_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=pooling_mode_max_tokens)

    # join BERT model and pooling to get the sentence transformer
    model = SentenceTransformer(modules=[sbert_worde_embedding_model, pooling_model])

    start_time = time.time()
    embeddings = model.encode(array, show_progress_bar=True)
    print("--- Embedding dimension {}".format(embeddings.shape[1]))
    print("--- %d Documnets encoded %s seconds ---" % (len(array), (time.time() - start_time)))
    return embeddings


def load_umap_and_cluster(embeddings,umap_model="umap_100000_6-neighbors_128-comps.pkl", **kwargs):
    """
    This function takes embeddings, loads the given pretrained UMAP model and computes the two dimensional projection
    $ready to be vizualized,
    :param embeddings:
    :param kwargs:
    :return:
    """
    # Load the Model back from file
    start_time = time.time()

    viz_model_path = "../models/bert-german-dbmdz-uncased-sentence-stsb/umap_viz_10_30-neighbors.pkl"
    dim_reduction_model_path = "../models/bert-german-dbmdz-uncased-sentence-stsb/"+umap_model

    with open(viz_model_path, 'rb') as file:
        fitted_umap_viz = pickle.load(file)

    with open(dim_reduction_model_path, 'rb') as file:
        fitted_umap_clustering = pickle.load(file)

    print("--- UMAP Loaded in %s seconds ---" % (time.time() - start_time))

    st = time.time()
    umap_data = fitted_umap_viz.transform(embeddings)
    print(">> Reducing dimensionality from {} to {} ...".format(embeddings.shape[1], str(128)),
          end="\r")

    umap_embeddings = fitted_umap_clustering.transform(embeddings)

    # Overriding default parameters
    params = {"min_cluster_size": 3, "min_samples": 3, "alpha": 0.78, "cluster_selection_epsilon": 0.1,
              "allow_single_cluster": True,
              "metric": 'euclidean',
              "cluster_selection_method": 'eom',
              "approx_min_span_tree": True}

    for (k, v) in kwargs.items():
        params[k] = v

    print(">> Clustering...", end="\r")
    clusters = HDBSCAN(**params).fit_predict(umap_embeddings)
    print(">> --- Done in {:.1f} seconds ---".format(time.time() - st), end="\r")
    print(">> Silhouette Coefficient: {}".format(metrics.silhouette_score(umap_embeddings, clusters)), end="\r")
    return umap_data, clusters


def cluster_and_reduce(embeddings, one_day=False, n_neighbors=15, n_components_clustering=384, **kwargs):
    st = time.time()
    umap_data = umap.UMAP(n_neighbors=n_neighbors, n_components=3, metric='cosine', random_state=0).fit_transform(
        embeddings)
    print(">> Reducing dimensionality from {} to {} ...".format(embeddings.shape[1], str(n_components_clustering)),
          end="\r")
    if len(embeddings) > n_components_clustering:
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components_clustering, random_state=0,
                                    metric='cosine').fit_transform(embeddings)
    else:
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components_clustering, random_state=0,
                                    metric='cosine', init="random").fit_transform(embeddings)

    params = {"min_cluster_size": 6, "min_samples": 3,
              "alpha": 0.88, "cluster_selection_epsilon": 0.11
        , "metric": 'euclidean',
              "cluster_selection_method": 'eom', "approx_min_span_tree": True}

    for (k, v) in kwargs.items():
        params[k] = v

    print(">> Clustering...", end="\r")
    clusters = HDBSCAN(**params).fit_predict(umap_embeddings)
    print(">> --- Done in {:.1f} seconds ---".format(time.time() - st), end="\r")
    print(">> Silhouette Coefficient: {}".format(metrics.silhouette_score(umap_embeddings, clusters)), end="\r")

    return umap_data, clusters


def scatter_plot(result, save_fig=False):
    if "labels" in result.columns.to_list():
        result["labels"] = result.labels.apply(str)
    elif "topic_number" in result:
        result["labels"] = result.topic_number.apply(str)

    fig = px.scatter(result, x="x", y="y", hover_name="headline", hover_data=["created_at"], color="labels",
                     opacity=0.8)
    fig.update_traces(marker=dict(size=9,
                                  line=dict(width=0.15,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig["layout"].pop("updatemenus")

    if save_fig:
        fig.update_layout(height=500)  # Dumping smaller images for convience
        fig.write_html("./tmp_scatter_plot.html")
    else:
        fig.update_layout(height=1000)
        fig.show()



################################

# Legacy functions below 

################################


def c_tf_idf(documents, m, ngram_range=(1, 1), remove_stop_words=True):
    if remove_stop_words:
        def remove_stop_words(doc):
            for sword in utils.STOPWORDS:
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


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes
