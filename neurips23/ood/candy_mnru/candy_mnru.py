import PyCANDYAlgo

import numpy as np

from neurips23.ood.base import BaseOODANN
from benchmark.datasets import DATASETS
from benchmark.dataset_io import download_accelerated
import os

class candy_mnru(BaseOODANN):
    def __init__(self, metric, index_params):
        self.indexkey="MNRU32"
        self.name = "candy_MNRU"
        self.ef=16

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        d = ds.d
        index = PyCANDYAlgo.index_factory_ip(d, self.indexkey)

        xb = ds.get_dataset()
        print("train")
        index.add(xb.shape[0], xb.flatten())
        self.index = index
        self.nb = ds.nb
        self.xb = xb

        return


    def query(self, X, k):


        querySize = X.shape[0]

        results = self.index.search(querySize, X.flatten(), k, self.ef)
        res = np.array(results).reshape(X.shape[0], k)

        self.res = res

    def set_query_arguments(self, query_args):
        if "ef" in query_args:
            self.ef = query_args['ef']
        else:
            self.ef = 16

    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.faissindex"