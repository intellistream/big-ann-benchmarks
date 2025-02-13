import PyCANDYAlgo
import os

from benchmark.algorithms.base import BaseANN

class BaseConcurrentANN(BaseANN): 
    def __init__(self, metric, index_params):
        self.indexkey= index_params['indexkey']
        self.metric = metric
        self.cc_result = None
        self.result = None
        
    def track(self):
        return "concurrent"

    def setup(self, dtype, max_pts, write_ratio, batch_size, num_threads, ndim):
        self.max_pts = max_pts
        self.index = PyCANDYAlgo.createIndex("ConcurrentIndex", ndim)
        
        cm = PyCANDYAlgo.ConfigMap()
        cm.edit("concurrentAlgoTag", self.indexkey)  
        cm.edit("vecDim", ndim)
        cm.edit("concurrentWriteRatio", write_ratio)
        cm.edit("concurrentBatchSize", batch_size)
        cm.edit("concurrentNumThreads", num_threads)

        metric_type = "L2" if self.metric == "euclidean" else "IP"
        cm.edit("metricType", metric_type)
        
        self.index.setConfig(cm)
        
    def cc_insert_and_query(self, t, qt, k):
        self.cc_result = self.index.ccInsertAndSearchTensor(t, qt, k)
    
    def query(self, qt, k):
        self.result = self.index.searchTensor(qt, k)
    
    def get_cc_results(self):
        return self.cc_result
    
    def get_result(self):
        return self.result
    
    def set_query_arguments(self, query_args):
        # TODO:
        pass
    
        
    
    
        