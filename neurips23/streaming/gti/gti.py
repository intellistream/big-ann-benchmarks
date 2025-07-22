import numpy as np
from gti_wrapper import GTIWrapper
from neurips23.streaming.base import BaseStreamingANN


class GTI_Index(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params
        self.name = "gti"
        self.is_built = False
        self.wrapper = None

        # Extract parameters with defaults
        self.capacity_up_i = index_params.get("capacity_up_i", 100)
        self.capacity_up_l = index_params.get("capacity_up_l", 100)
        self.m = index_params.get("m", 4)

        # Query parameters
        self.query_L = 60  # Default L value for search

    def setup(self, dtype, max_pts, ndim):
        """Setup the GTI index with given parameters"""
        self.ndim = ndim
        self.max_pts = max_pts
        self.dtype = dtype

        # Initialize the wrapper
        self.index = GTIWrapper()
        self.index.setup(max_pts, ndim, self.capacity_up_i, self.capacity_up_l, self.m)
        print(f"GTI setup: max_pts={max_pts}, ndim={ndim}, capacity_up_i={self.capacity_up_i}, "
              f"capacity_up_l={self.capacity_up_l}, m={self.m}")

    def insert(self, X, ids):
        """Insert vectors into the GTI index"""
        # Convert to appropriate types
        # X = X.astype(np.float32)
        X = np.ascontiguousarray(X.astype(np.float32))
        # print(X.flags['C_CONTIGUOUS'])
        # print(X.shape)
        ids = ids.astype(np.int32)
        # print(ids.dtype == np.int32)
        # print(len(np.unique(ids)) == len(ids))
        if not self.is_built:
            # First insertion - build the index
            self.index.build(X, ids, self.capacity_up_i, self.capacity_up_l, self.m)
            self.is_built = True
            print("GTI index built successfully")
        else:
            # Subsequent insertions
            self.index.insert(X, ids)
            # self.index.debug_info()

    def delete(self, ids):
        """Delete vectors from the GTI index by their IDs"""
        if not self.is_built:
            print("Warning: Cannot delete from unbuilt index")
            return

        ids = ids.astype(np.int32)
        self.index.remove(ids)
        # self.index.debug_info()
        print("Vectors deleted successfully")

    def query(self, X, k):
        """Query the GTI index for k nearest neighbors"""
        if not self.is_built:
            raise RuntimeError("Index must be built before querying")

        # X = X.astype(np.float32)
        X = np.ascontiguousarray(X.astype(np.float32))
        n = X.shape[0]

        # Perform the query
        results, distances = self.index.query(X, k, self.query_L)
        # results, distances = self.index.query(self.temp_data[10:20], k, self.query_L)
        # Store results for later access
        # print(results)
        self.res = results


    def set_query_arguments(self, query_args):
        """Set query-time parameters"""
        self.query_L = query_args.get("L", 60)

    def index_name(self, name):
        """Generate index file name"""
        return f"data/{name}.gti"

    def get_memory_usage(self):
        """Get memory usage information (if available)"""
        if self.index:
            return {"index_size": self.index.size()}
        return {"index_size": 0}
