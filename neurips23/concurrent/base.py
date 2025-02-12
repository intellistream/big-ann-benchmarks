import PyCANDYAlgo
from PyCANDYAlgo import ConfigMap

from benchmark.algorithms.base import BaseANN

class BaseConcurrentANN(BaseANN): 
    def __init__(self, metric, index_params):
        self.index = PyCANDYAlgo.ConcurrentIndex()
        self.indexkey= index_params['indexkey']
        self.metric = metric
        self._setup_index()
        
    def track(self):
        return "concurrent"

    def _setup_index(self, ndim):
        cm = ConfigMap()
        cm.edit("concurrentAlgoTag", self.indexkey)  
        cm.edit("vecDim", ndim)
        cm.edit("concurrentWriteRatio", self.config.get("write_ratio", 0.5))
        cm.edit("concurrentBatchSize", self.config.get("batch_size", 100))
        cm.edit("concurrentNumThreads", self.config.get("num_threads", 1))

        metric_type = "L2" if self.metric == "euclidean" else "IP"
        cm.edit("metricType", metric_type)
        self.index.setConfig(cm)
        
    def cc_insert_and_query(self, t, qt, k):
        return self.index.ccInsertAndSearchTensor(t, qt, k)
    
    def query(self, qt, k):
        return self.index.searchTensor(qt, k)
    
        
    
    
        