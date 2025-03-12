from neurips23.concurrent.base import BaseConcurrentANN

class pyanns(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params)
        self.name = "pyanns"
        if (index_params.get("R") == None):
            print("Error: missing parameter R")
            return
        if (index_params.get("L") == None):
            print("Error: missing parameter L")
            return
        self.cm.edit("concurrentAlgoTag", "Pyanns") 
        self.cm.edit("R", index_params["R"])
        self.cm.edit("L", index_params["L"])
        
    def set_query_arguments(self, query_args):
        # TODO:
        pass 