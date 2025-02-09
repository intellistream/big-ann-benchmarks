import numpy as np
from numpy import typing as npt

from neurips23.concurrent.base import BaseConcurrentANN
from neurips23.concurrent.diskann.diskann import diskann as diskann_concurrent

class diskann(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__([diskann_concurrent(metric, index_params)], metric, index_params)
        self.metric = metric
        self.name = self.workers[0].my_index_algo.index_name()

    def set_query_arguments(self, query_args):
        self.workers[0].my_index_algo.set_query_arguments(query_args)


