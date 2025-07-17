import numpy as np
from neurips23.streaming.base import BaseStreamingANN
import ctypes
# 先显式加载 libSPTAGLib.so
ctypes.CDLL("/usr/local/lib/libSPTAGLib.so")
# from neurips23.streaming.spfresh.spfresh_init import spfresh_py  # 对应的是 pybind11 导出的模块
import spfresh_py

class SPFresh(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params
        self.name = "spfresh"
        self.is_built = False
        self.index = None

    def setup(self, dtype, max_pts, ndim):
        self.ndim = ndim
        self.dtype = dtype
        self.max_pts = max_pts

        # 提取参数
        self.index_directory = self.index_params.get("index_directory", "./data/spfresh_index")
        self.ssd_build_threads = self.index_params.get("ssd_build_threads", 16)
        self.insert_threads = self.index_params.get("insert_threads", 16)
        self.delete_threads = self.index_params.get("delete_threads", 16)
        # self.search_threads = self.index_params.get("search_threads", 16)
        self.normalized = self.index_params.get("normalized", True)


        self.vector_value_type = spfresh_py.VectorValueType.Float
        # 创建 SPFreshIndex 实例
        self.index = spfresh_py.SPFreshIndex(ndim, self.vector_value_type)
    def insert(self, X, ids):

        # X = X.astype(np.float32)
        X = np.ascontiguousarray(X, dtype=np.float32)
        # ids = ids.astype(np.uint32)
        ids = np.ascontiguousarray(ids, dtype=np.uint32)

        if not self.is_built:
            # print("test")
            # 首次构建索引
            self.index.build(
                X,
                self.index_directory,
                self.ssd_build_threads,
                self.normalized
            )
            # index = spfresh_py.SPFreshIndex(128, spfresh_py.VectorValueType.Float)
            #
            # # 2. 生成测试数据
            # print("2. 生成测试数据...")
            # vectors = np.random.randn(100000, 128).astype(np.float32)
            #
            # # 3. 构建索引
            # print("3. 构建索引...")
            # index.build(vectors, "./spfresh_index", ssd_build_threads=2, normalize=True)
            self.is_built = True
        else:
            # 后续插入
            self.index.insert(X, ids.tolist(), self.insert_threads)

    def delete(self, ids):
        if not self.is_built:
            raise RuntimeError("Index not built yet")

        ids = ids.astype(np.uint32)
        self.index.remove(ids.tolist(), self.delete_threads)

    def query(self, X, k):
        if not self.is_built:
            raise RuntimeError("Index not built yet")

        # X = X.astype(np.float32)
        X = np.ascontiguousarray(X, dtype=np.float32)

        # 调用搜索接口
        results_list = self.index.search(X, k, self.search_threads)

        # 转换结果格式
        n = X.shape[0]
        results = np.zeros((n, k), dtype=np.uint32)

        for i, result in enumerate(results_list):
            # 处理结果长度不足 k 的情况
            actual_k = min(len(result), k)
            results[i, :actual_k] = result[:actual_k]
            # 如果结果不足 k 个，用 -1 填充
            if actual_k < k:
                results[i, actual_k:] = -1

        self.res = results

    def set_query_arguments(self, query_args):
        """设置查询参数"""
        # SPFresh 的查询参数可以在这里设置
        # 根据 C++ 代码，主要是线程数参数
        self.search_threads = query_args.get("search_threads", 16)

    def index_name(self, name):
        return f"data/{name}.spfresh"
