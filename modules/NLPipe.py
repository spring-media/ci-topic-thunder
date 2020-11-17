import math, pdb, os
import torch, time
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans as KMeans
from sentence_transformers import SentenceTransformer, util, models
from typing import List, Union
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from box import Box
from kneed import KneeLocator

from modules.utils import preprocess_articles_for_bert, load_labeled_data
from modules.modeling import scatter_plot,bar_plot
from modules.modeling import _preload_umap_reduce,_new_umap_reduce


# https://stackoverflow.com/a/7590709/362790
def chain(device_in=None, keep=None, together=False):
    """
    Decorator for chaining methods in NLPipe like NLPipe(x, y).embed().normalize().cosine()
    When you want the output at any step, call .value(). It will retain its intermediate step
    so you can continue chaining later, and call subsequent .value()
    :param device_in: gpu|cpu|None. What device does this chain-step expect its values from?
    :param keep: keep data across chains by this key. chain.data[key]
    :param together: whether to process x & y as a whole, then split back apart (eg, tf-idf)
    """

    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            # Place x,y on device this chain method expects
            x, y = self.get_values(device_in)
            x_y = (self._join(x, y),) if together else (x, y)
            res = fn(self, *x_y, *args, **kwargs)

            data = self.data
            if keep:
                data[keep] = res[-1]
                res = res[0]

            if together:
                res = self._split(res, x, y)

            # Always maintain [x, y] for consistency
            if type(res) != list: res = [res, None]
            # Save intermediate result, and chained methods can continue

            return self.__class__(*res, last_fn=fn.__name__, data=data)

        return wrapper

    return decorator


class NLPipe:
    """
    Various similarity helper functions.

    * NLP methods: cleantext, tf_idf, embed (via sentence_transformers), etc
    Call like Similars(sentences).embed() or Similars(lhs, rhs).cleantext().tfidf()

    * Similarity methods: normalize, cosine, kmeans, agglomorative, etc
    Clustering: Similars(x, y).normalize().cluster(algo='agglomorative')
    Similarity: Similars(x).normalize.cosine()  (then sort low to high)

    Takes x, y. If y is provided, then we're comparing x to y. If y is None, then operations
    are pairwise on x (x compared to x).
    """

    def __init__(
            self,
            x: Union[List[str], np.ndarray],
            y: Union[List[str], np.ndarray] = None,
            last_fn=None,
            data=Box(default_box=True, default_box_attr=None),**kwargs
    ):
        self.result = [x, y]
        self.last_fn = last_fn
        self.data = data
        # Apply mean pooling to get one fixed sized sentence vector
        word_embedding_model = models.Transformer('T-Systems-onsite/bert-german-dbmdz-uncased-sentence-stsb',max_seq_length=250)

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        # join BERT model and pooling to get the sentence transformer
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def value(self):
        x, y = self.get_values('cpu', unsqueeze=False)
        if y is None: return x
        return [x, y]

    def get_values(self, device=None, unsqueeze=True):
        x, y = self.result
        if device is None:
            return x, y
        if device == 'gpu':
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            if y is not None and not torch.is_tensor(y):
                y = torch.tensor(y)
        elif device == 'cpu':
            if torch.is_tensor(x):
                x = x.cpu().numpy()
            if y is not None and torch.is_tensor(y):
                y = y.cpu().numpy()
        else:
            raise Exception("Device must be (gpu|cpu)")
        if unsqueeze:
            x, y = self._unsqueeze(x), self._unsqueeze(y)
        return x, y

    @staticmethod
    def _join(x, y):
        if y is None: return x
        if type(x) == list: return x + y
        if type(x) == np.ndarray: return np.vstack([x, y])
        if torch.is_tensor(x): return torch.cat((x, y), 0)

    @staticmethod
    def _split(joined, x, y):
        if y is None: return [joined, None]
        at = len(x) if type(x) == list else x.shape[0]
        return [joined[:at], joined[at:]]

    @staticmethod
    def _unsqueeze(t):
        if t is None: return t
        if len(t.shape) > 1: return t
        return t.unsqueeze(0)

    @chain(together=True)
    def embed(self, both: List[str], batch_size=32):
        return self.model.encode(
            both,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )

    @chain(together=True)
    def reduce_dim(self, both: List[float],
                   model_path="../models/bert-german-dbmdz-uncased-sentence-stsb/umap_100k_6-neighbors_384-comps.pkl"):
        return _preload_umap_reduce(both,
                                    model=model_path)

    @chain(together=True)
    def reduce_dim_fresh(self,both: List[float],params):

        return _new_umap_reduce(embeddings=both,args=params)

    @chain(together=True, keep='headlines')
    def _store_headlines(self, df):
        return [df.headline.values]

    @chain(together=True, keep='created_at')
    def _store_created_at(self, df):
        return [df.created_at.values]

    @chain(together=True, keep="pdf")
    def cleantext(self, both: pd.DataFrame, **kwargs):
        return [preprocess_articles_for_bert(both, **kwargs)]

    @chain(together=True)
    def pca(self, both, **kwargs):
        return PCA(**kwargs).fit_transform(both)

    @chain(together=True)
    def tf_idf(self, both):
        return TfidfVectorizer().fit_transform(both)

    @chain(device_in='gpu', together=True)
    def normalize(self, both):
        return both / both.norm(dim=1)[:, None]

    def _sims_by_clust(self, x, top_k, fn):
        assert torch.is_tensor(
            x), "_sims_by_clust written assuming GPU in, looks like I was wrong & got a CPU val; fix this"
        assert top_k, "top_k must be specified if using clusters in similarity functions"
        labels = self.data.labels[0]
        res = []
        for l in range(labels.max()):
            mask = (labels == l)
            if mask.sum() == 0: continue
            k = math.ceil(mask.sum() / top_k)
            x_ = x[mask].mean(0)
            r = fn(x_, k)
            res.append(torch.cat((r.values, r.indices), 1))
        return torch.stack(res)

    def _cosine(self, x, y, abs=False, top_k=None):
        if y is None: y = x
        sim = torch.mm(x, y.T)

        if abs:
            # See https://stackoverflow.com/a/63532174/362790 for the various options
            # print("sim.min=", sim.min(), "sim.max=", sim.max())
            eps = np.finfo(float).eps
            dist = sim.clamp(-1 + eps, 1 - eps).acos() / np.pi
            # dist = (sim - 1).abs()  # <-- used this before, ends up in 0-2 range
            # dist = 1 - (sim + 1) / 2
        else:
            dist = 1. - sim
        if top_k is None: return dist
        return torch.topk(dist, min(top_k, dist.shape[1] - 1), dim=1, largest=False, sorted=False)

    @chain(device_in='gpu')
    def cosine(self, x, y, abs=False, top_k=None):
        """
        :param abs: Hierarchical clustering wants [0 1], and dists.sort_by(0->1), but cosine is [-1 1]. Set True to
            ensure cosine>0. Only needed currently for agglomorative()
        :param top_k: only return top-k smallest distances. If you cluster() before this cosine(), top_k is required.
            It will return (n_docs_in_cluster/top_k) per cluster.
        """
        if self.last_fn != 'cluster':
            res = self._cosine(x, y, abs=abs)
            if not top_k: return res

        def fn(x_, k):
            return self._cosine(x_, y, abs=abs, top_k=k)

        return self._sims_by_clust(x, top_k, fn)

    @chain(device_in='cpu')
    def cdist(self, x, y, **kwargs):
        return cdist(x, y, **kwargs)

    @chain(device_in='cpu')
    def ann(self, x, y, y_from_file=None, top_k=None):
        """
        Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_hnswlib.py
        Finds top-k similar y similar embeddings to x.
        Make sure you call .normalize() before this step!
        :param y_from_file: if provided, will attempt to load from this path. If fails, will train index & save to
            this path, to be loaded next time
        :param top_k: how many results per x-row to return? If 1, just find closest match per row.
            cluster-mean
        """
        try:
            import hnswlib
        except:
            raise Exception("hnswlib not installed; install it manually yourself via `pip install hnswlib`")
        if y is None: raise Exception("y required; it's the index you query")
        if y_from_file and os.path.exists(y_from_file):
            index = hnswlib.Index(space='cosine', dim=x.shape[1])
            index.load_index(y_from_file)
        else:
            # Defining our hnswlib index
            # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
            index = hnswlib.Index(space='cosine', dim=y.shape[1])

            ### Create the HNSWLIB index
            print("Start creating HNSWLIB index")
            # UKPLab tutorial used M=16, but https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md suggests 64
            # for word embeddings (though these are sentence-embeddings)
            index.init_index(max_elements=y.shape[0], ef_construction=200, M=64)

            # Then we train the index to find a suitable clustering
            index.add_items(y, np.arange(y.shape[0]))
            index.save_index(y_from_file)

        # Controlling the recall by setting ef:
        # ef = 50
        ef = max(top_k + 1, min(1000, index.get_max_elements()))
        index.set_ef(ef)  # ef should always be > top_k_hits

        def fn(x_, k):
            # We use hnswlib knn_query method to find the top_k_hits
            return index.knn_query(x_, k)

        if self.data.labels is None:
            return fn(x, top_k)
        return self._sims_by_clust(x, top_k, fn)

    @chain(keep='labels')
    def cluster_hdbscan(self, x, y, **kwargs):
        both = self._join(x, y)
        n = both.shape[0]
        st = time.time()

        # Overriding default parameters
        params = {"min_cluster_size": 3, "min_samples": 3, "alpha": 1.0, "cluster_selection_epsilon": 0.14,
                  "allow_single_cluster": True,
                  "metric": 'euclidean',
                  "cluster_selection_method": 'eom',
                  "approx_min_span_tree": True}

        for (k, v) in kwargs.items():
            params[k] = v
        print(params)
        cluster_labels = HDBSCAN(**params).fit_predict(x)
        print(">> --- Done in {:.1f} seconds ---".format(time.time() - st))
        return [cluster_labels]


def main():
    # df = pd.read_csv("../data/welt_articles.csv", index_col=0)
    # df = df[df.department != "mediathek"]
    # df = df.rename(columns={"date": "created_at"})
    # df = df.sort_values(by='created_at').tail(500)
    # df.created_at = pd.to_datetime(df.created_at, dayfirst=True)
    # df.created_at=df.created_at.apply(lambda x: x.date()).apply(str)
    # df = df.rename(columns={"title": "headline"})
    #
    # df.text = df.headline + ". " + df.text
    # nlpipe = NLPipe(df)
    # nlpipe._store_headlines()
    # nlpipe._store_created_at()
    # embedded = nlpipe.cleantext().embed(batch_size=50)
    #
    # cluster_results = embedded.reduce_dim().cluster_hdbscan(cluster_selection_method="leaf")
    #
    # res = pd.DataFrame(embedded.reduce_dim(
    #     model_path="../models/bert-german-dbmdz-uncased-sentence-stsb/umap_viz_100_19-neighbors_0.01-min-dist.pkl").value(),
    #                    columns=['x', 'y'])
    #
    # res['labels'] = cluster_results.value()
    # res['headline'] = nlpipe.data.headlines
    # res['created_at'] = nlpipe.data.created_at
    # #scatter_plot(res,animation_frame="created_at")
    # bar_plot(res)
    print("ALL DONE ")

if __name__ == "__main__":
    main()
