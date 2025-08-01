import numpy as np
from neurips23.streaming.base import BaseStreamingANN
import dynagraph  # 对应的是 pybind11 导出的模块

class DynaGraph(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params
        self.name = "dynagraph"
        self.is_built = False

    def setup(self, dtype, max_pts, ndim):
        self.ndim = ndim
        self.index = dynagraph.Index()
        self.max_pts = max_pts

        # 提取参数
        alpha = self.index_params.get("alpha", 1.2)
        coef_L = self.index_params.get("coef_L", 100)
        coef_R = self.index_params.get("coef_R", 64)
        # 增大batch_size避免小数据集触发批处理问题
        batch_size = self.index_params.get("batch_size", 10000)  
        self.insert_thread_count = self.index_params.get("insert_thread_count", 1)
        self.search_thread_count = self.index_params.get("search_thread_count", 1)
        
        self.index.setup(max_pts, ndim, alpha, coef_L, coef_R, batch_size)

    def insert(self, X, ids):
        # ids 是一维 np.array，X 是二维 np.array
        X = X.astype(np.float32)
        # DynaGraph使用uint64_t，不需要+1偏移
        ids = ids.astype(np.uint64)
        
        if not self.is_built:
            self.index.build(X, X.shape[0], ids.tolist())
            self.is_built = True
        else:
            # 增量插入 - 使用insert_point方法确保向量数据被正确存储
            for i in range(len(ids)):
                point = X[i]  # 1D数组，符合pybind11接口要求
                id_val = ids[i]
                success = self.index.insert_point(point, int(id_val))
                if not success:
                    print(f"Warning: Failed to insert point {id_val}")

    def delete(self, ids):
        # DynaGraph使用uint64_t，不需要+1偏移
        ids = ids.astype(np.uint64)
        self.index.remove(ids.tolist())

    def query(self, X, k):
        n = X.shape[0]
        results = np.zeros((n, k), dtype=np.uint64)
        dists = np.zeros((n, k), dtype=np.float32)

        for i in range(n):
            query_vec = X[i].astype(np.float32)
            tags, distances = self.index.query(query_vec, k)
            # DynaGraph不需要-1偏移
            tags_array = np.array(tags, dtype=np.uint64)
            distances_array = np.array(distances, dtype=np.float32)
            
            # 处理返回结果可能少于k的情况
            actual_k = min(len(tags_array), k)
            if actual_k > 0:
                results[i, :actual_k] = tags_array[:actual_k]
                dists[i, :actual_k] = distances_array[:actual_k]
                # 填充剩余位置
                if actual_k < k:
                    results[i, actual_k:] = tags_array[0] if len(tags_array) > 0 else 0
                    dists[i, actual_k:] = float('inf')

        # 可选：使用批量查询进行优化
        # tags, distances = self.index.batch_query(X, k, self.search_thread_count)
        # results = tags.astype(np.uint64)
        # results = results.reshape(n, k)
        
        self.res = results
        return results

    def set_query_arguments(self, query_args):
        # DynaGraph的查询参数可以在这里扩展
        self.query_L = query_args.get("L", 128)  # 暂时未使用，可扩展进 C++ 参数

    def index_name(self, name):
        return f"data/{name}.dynagraphindex"
