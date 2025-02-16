import PyCANDYAlgo
import os

from benchmark.algorithms.base import BaseANN

CC_RESULT_DIR = "results/concurrent"

class BaseConcurrentANN(BaseANN): 
    def __init__(self, metric, index_params):
        self.index = None
        self.indexkey= index_params['indexkey']
        self.metric = metric
        self.cc_config = {}
        self.cc_result = None
        self.result = None
        self.cc_res_filename = ""
        
    def track(self):
        return "concurrent"

    def setup(self, dtype, max_pts, cc_config, ndim):
        self.max_pts = max_pts
        self.index = PyCANDYAlgo.createIndex("Concurrent", ndim)
        
        cm = PyCANDYAlgo.ConfigMap()
        cm.edit("concurrentAlgoTag", self.indexkey)  
        cm.edit("vecDim", ndim)
        cm.edit("ccWriteRatio", cc_config['write_ratio'])
        cm.edit("ccBatchSize", cc_config['batch_size'])
        cm.edit("ccNumThreads", cc_config['num_threads'])
        cm.edit("ccRandomMode", cc_config['random_mode'])

        metric_type = "L2" if self.metric == "euclidean" else "IP"
        cm.edit("metricType", metric_type)
        
        self.cc_config = cc_config
        self.res_filename = f"batch{cc_config['batch_size']}_w{cc_config['write_ratio']}"
        
        self.index.setConfig(cm)
        
    def cc_insert_and_query(self, t, qt, k):
        return self.index.ccInsertAndSearchTensor(t, qt, k)
    
    def query(self, qt, k):
        self.result = self.index.searchTensor(qt, k)
    
    def get_cc_results(self):
        if not self.index.ccSaveReesultAsFile(self.cc_res_filename):
            raise ValueError("Result save failed")
        return self.cc_res_filename 
    
    def get_result(self):
        return self.result
    
    def get_cc_config(self):
        return self.cc_config
        
    def set_query_arguments(self, query_args):
        raise NotImplementedError()
    
        
    
    
        