from neurips23.concurrent.base import BaseConcurrentANN

class faiss_HNSW(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params)
        self.name = "faiss_HNSW"
        self.cm.edit("concurrentAlgoTag", "faiss") 
        self.cm.edit("faissIndexTag", "HNSW")
        self.cm.edit("maxConnection", index_params["maxConnection"])
        
    def set_query_arguments(self, query_args):
        # TODO:
        pass 