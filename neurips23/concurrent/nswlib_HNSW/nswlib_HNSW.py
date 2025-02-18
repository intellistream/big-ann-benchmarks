from neurips23.concurrent.base import BaseConcurrentANN

class nswlib_HNSW(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params)
        self.name = "nswlib_HNSW"
        self.cm.edit("concurrentAlgoTag", "NSWlibHNSW") 
        self.cm.edit("maxConnection", index_params["maxConnection"])
        self.cm.edit("efConstruction", index_params["efConstruction"])
        
    def set_query_arguments(self, query_args):
        # TODO:
        pass 