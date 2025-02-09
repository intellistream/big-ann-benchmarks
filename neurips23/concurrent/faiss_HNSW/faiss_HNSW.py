import numpy as np
from numpy import typing as npt

from neurips23.concurrent.base import BaseConcurrentANN
from neurips23.concurrent.faiss_HNSW.faiss_HNSW import faiss_HNSW as faiss_HNSW_concurrent

class faiss_HNSW(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__([faiss_HNSW_concurrent(metric, index_params)], metric, index_params)
        self.metric = metric
        self.indexkey=self.workers[0].my_index_algo.indexkey
        self.name = self.workers[0].my_index_algo.name

    def set_query_arguments(self, query_args):
        self.workers[0].my_index_algo.set_query_arguments(query_args)


