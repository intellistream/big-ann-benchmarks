import numpy as np
from neurips23.streaming.base import BaseStreamingANN
import freshdiskann  # 对应的是 pybind11 导出的模块


class FreshDiskANN(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params
        self.name = "freshdiskann"
        self.is_built = False

    def setup(self, dtype, max_pts, ndim):
        self.ndim = ndim
        self.index = freshdiskann.Index()
        self.max_pts = max_pts

        # 提取参数
        R = self.index_params.get("R", 64)
        L = self.index_params.get("L", 100)
        num_threads = self.index_params.get("num_threads", 32)
        self.insert_thread_count = self.index_params.get("insert_thread_count", 32)
        self.search_thread_count = self.index_params.get("search_thread_count", 32)

        # 设置索引参数
        self.index.setup(max_pts, ndim, R, L, num_threads)

    def insert(self, X, ids):
        # ids 是一维 np.array，X 是二维 np.array
        X = X.astype(np.uint8)
        ids = ids.astype(np.uint32)

        if not self.is_built:
            # 首次构建索引
            self.index.build(X, X.shape[0], ids.tolist())
            self.is_built = True
        else:
            # 后续并发插入
            self.index.insert_concurrent(X, ids, self.insert_thread_count)

    def delete(self, ids):
        ids = ids.astype(np.uint32)
        self.index.remove(ids.tolist())

    def query(self, X, k):
        n = X.shape[0]
        results = np.zeros((n, k), dtype=np.uint32)
        dists = np.zeros((n, k), dtype=np.float32)

        # 使用批量查询以提高效率
        if self.search_thread_count > 1 and n > 1:
            # 如果支持批量查询且使用多线程
            X_query = X.astype(np.uint8)
            tags, distances = self.index.batch_query(X_query, k, self.search_thread_count)
            results = tags.astype(np.uint32)
            results = results.reshape(n, k)
            dists = distances.reshape(n, k)
        else:
            # 逐个查询
            for i in range(n):
                query_vec = X[i].astype(np.uint8)
                tags, distances = self.index.query(query_vec, k, getattr(self, 'query_L', 10))
                results[i] = np.array(tags, dtype=np.uint32)
                dists[i] = distances

        self.res = results

    def set_query_arguments(self, query_args):
        self.query_L = query_args.get("L", 20)  # 查询时使用的L参数

    def index_name(self, name):
        return f"data/{name}.freshdiskannindex"

    def final_merge(self):
        """执行最终合并操作"""
        if self.is_built:
            self.index.final_merge()

    def save_index(self, filepath):
        """保存索引到文件"""
        if self.is_built:
            self.index.save_index(filepath)

    def load_index(self, filepath):
        """从文件加载索引"""
        self.index.load_index(filepath)
        self.is_built = True

    def get_stats(self):
        """获取索引统计信息"""
        if self.is_built:
            return self.index.get_stats()
        return {}

    def insert_single(self, point, tag):
        """插入单个点（辅助方法）"""
        if not self.is_built:
            raise RuntimeError("Index not built yet")

        point = point.astype(np.uint8)
        tag = np.uint32(tag) + 1  # 转换为从1开始的标签
        return self.index.insert(point, tag)