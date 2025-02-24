from neurips23.concurrent.base import BaseConcurrentANN

class diskann(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params)
        self.name = "nswlib_HNSW"
        self.cm.edit("concurrentAlgoTag", "NSWlibHNSW") 
        self.cm.edit("maxConnection", index_params["maxConnection"])
        self.cm.edit("efConstruction", index_params["efConstruction"])
        
        self.name = "diskann"
        if (index_params.get("R") == None):
            print("Error: missing parameter R")
            return
        if (index_params.get("L") == None):
            print("Error: missing parameter L")
            return
        self.cm.edit("concurrentAlgoTag", "diskann") 
        self.cm.edit("R", index_params["R"])
        self.cm.edit("L", index_params["L"])
        self.cm.edit("insert_threads", index_params["insert_threads"])
        self.cm.edit("consolidate_threads", index_params["consolidate_threads"])
        
    def set_query_arguments(self, query_args):
        # TODO:
        pass 