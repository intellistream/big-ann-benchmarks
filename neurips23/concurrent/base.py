import PyCANDYAlgo
import torch

from benchmark.algorithms.base import BaseANN

CC_RESULT_DIR = "results/concurrent"

class BaseConcurrentANN(BaseANN): 
    def __init__(self, metric, index_params):
        self.index = None
        self.metric = metric
        self.cc_result = None
        self.result = None
        self.cc_config = {}
        self.cc_res_filename = ""
        
        self.cm = PyCANDYAlgo.ConfigMap()
        
    def track(self):
        return "concurrent"

    def setup(self, dtype, max_pts, cc_config, ndim):
        self.max_pts = max_pts
        self.index = PyCANDYAlgo.createIndex("Concurrent", ndim)
        self.cm.edit("vecDim", ndim)
        self.cm.edit("ccWriteRatio", cc_config['write_ratio'])
        self.cm.edit("ccBatchSize", cc_config['batch_size'])
        self.cm.edit("ccNumThreads", cc_config['num_threads'])
        self.cm.edit("ccRandomMode", cc_config['random_mode'])
        self.cm.edit("maxElements", max_pts)
        
        metric_type = "L2" if self.metric == "euclidean" else "IP"
        self.cm.edit("metricType", metric_type)
        
        self.index.setConfig(self.cm)
        
        ratio_suffix = f"w{int(cc_config['write_ratio'] * 100)}r{int((1 - cc_config['write_ratio']) * 100)}"
        self.cc_res_filename = f"batch{cc_config['batch_size']}_{ratio_suffix}"
        self.cc_config = cc_config
        
    def initial(self, t):
        tensor = torch.from_numpy(t)
        return self.index.insertTensor(tensor)
        
    def cc_insert_and_query(self, t, qt, k):
        tensor = torch.from_numpy(t)
        qtensor = torch.from_numpy(qt)
        return self.index.ccInsertAndSearchTensor(tensor, qtensor, k)
    
    def query(self, qt, k):
        qtensor = torch.from_numpy(qt)
        self.result = self.index.searchTensor(qtensor, k)
    
    def get_cc_results(self):
        print(self.cc_res_filename)
        if not self.index.ccSaveResultAsFile(self.cc_res_filename):
            raise ValueError("Result save failed")
        return self.cc_res_filename 
    
    def get_results(self):
        return self.result
    
    def get_cc_config(self):
        return self.cc_config
        
    def set_query_arguments(self, query_args):
        raise NotImplementedError()
    
        