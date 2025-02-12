import PyCANDYAlgo
import os

from benchmark.algorithms.base import BaseANN

class BaseConcurrentANN(BaseANN): 
    def __init__(self, metric, index_params):
        self.indexkey= index_params['indexkey']
        self.batch_size = index_params['batch_size']
        self.write_ratio = index_params['write_ratio']
        
        if 'num_threads' in index_params:
            self.num_threads = os.cpu_count()
        else:
            self.num_threads = index_params['num_threads']
        
        self.metric = metric
        
    def track(self):
        return "concurrent"

    def setup(self, dtype, max_pts, ndim):
        self.index = PyCANDYAlgo.createIndex("ConcurrentIndex", ndim)
        
        cm = PyCANDYAlgo.ConfigMap()
        cm.edit("concurrentAlgoTag", self.indexkey)  
        cm.edit("vecDim", ndim)
        cm.edit("concurrentWriteRatio", self.write_ratio)
        cm.edit("concurrentBatchSize", self.batch_size)
        cm.edit("concurrentNumThreads", self.num_threads)

        metric_type = "L2" if self.metric == "euclidean" else "IP"
        cm.edit("metricType", metric_type)
        
        self.index.setConfig(cm)
        
    def cc_insert_and_query(self, t, qt, k):
        return self.index.ccInsertAndSearchTensor(t, qt, k)
    
    def query(self, qt, k):
        return self.index.searchTensor(qt, k)
    
        
    
    
        