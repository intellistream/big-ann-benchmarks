import PyCANDYAlgo
from PyCANDYAlgo import ConfigMap
from neurips23.concurrent.base import BaseConcurrentANN

class faiss_hnsw(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params)
        