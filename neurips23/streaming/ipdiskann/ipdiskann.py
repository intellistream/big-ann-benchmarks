import numpy as np
from neurips23.streaming.base import BaseStreamingANN
import ipdiskann

class IPdiskann(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params
        # self.indexkey = index_params.get("indexkey", "NA")
        self.name = "ipdiskann"
        self.is_built = False

    def setup(self, dtype, max_pts, ndim):
        self.ndim = ndim
        self.index = ipdiskann.Index()
        self.max_pts = max_pts


        R = self.index_params.get("R", 64)
        L = self.index_params.get("L", 100)
        num_threads = self.index_params.get("num_threads", 1)
        self.insert_thread_count = self.index_params.get("insert_thread_count", 1)
        self.search_thread_count = self.index_params.get("search_thread_count", 1)
        self.index.setup(max_pts, ndim, R, L, num_threads)

    def insert(self, X, ids):

        X = X.astype(np.float32)
        ids = ids.astype(np.uint32) + 1
        # print(ids)
        if not self.is_built:
            self.index.build(X, X.shape[0], ids.tolist())
            # self.index.build(X, self.max_pts, ids.tolist())
            self.is_built = True
        else:
            # for i in range(len(ids)):
            #     tag = int(ids[i])
            #     point = X[i]
            #     success = self.index.insert(point, tag)
            #     if not success:
            #         print(f"[Insert] Failed to insert tag {tag}")
            self.index.insert_concurrent(X, ids, self.insert_thread_count)

    def delete(self, ids):
        ids = ids.astype(np.uint32) + 1
        # print(ids)
        self.index.remove(ids.tolist())

    def query(self, X, k):
        n = X.shape[0]
        results = np.zeros((n, k), dtype=np.uint32)
        dists = np.zeros((n, k), dtype=np.float32)

        for i in range(n):
            query_vec = X[i].astype(np.float32)
            tags, distances = self.index.query(query_vec, k)
            results[i] = np.array(tags, dtype=np.uint32) - 1
            dists[i] = distances
        self.res = results

    def set_query_arguments(self, query_args):
        self.query_L = query_args.get("L", 128)  # 暂时未使用，可扩展进 C++ 参数

    def index_name(self, name):
        return f"data/{name}.ipdiskannindex"
