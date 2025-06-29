import numpy as np
from gti_module import GTI, Objects
from neurips23.streaming.base import BaseStreamingANN


class GTI_Index(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.name = "gti"
        self.indexkey = "GTI"

        # 参数配置
        self.params = {
            'capacity_up_i': 100,
            'capacity_up_l': 100,
            'm': 4,
            'L': 10,
            'K': 3
        }
        self.params.update(index_params)

        self.index = None
        self.dim = 0
        self.id_to_idx = {}  # ID到存储索引的映射
        self.idx_to_id = {}  # 存储索引到ID的映射
        self.next_idx = 0  # 下一个可用的存储索引
        self.is_built = False

    def setup(self, dtype, max_pts, ndim):
        self.dim = ndim
        self.index = GTI()
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        self.next_idx = 0
        self.is_built = False

    def insert(self, X, ids):
        # 存储数据并创建映射
        new_vectors = []
        for i, vector_id in enumerate(ids):
            vector = X[i].astype(np.float32).copy()
            self.id_to_idx[vector_id] = self.next_idx
            self.idx_to_id[self.next_idx] = vector_id
            self.next_idx += 1
            new_vectors.append(vector)

        # 转换为Objects
        new_data = Objects()
        new_array = np.vstack(new_vectors).astype(np.float32)
        new_data.setData(new_array)

        if not self.is_built:
            # 首次构建
            self.index.buildGTI(
                self.params['capacity_up_i'],
                self.params['capacity_up_l'],
                self.params['m'],
                new_data
            )
            self.is_built = True
        else:
            # 增量插入
            self.index.insertGTI(new_data)

    def delete(self, ids):
        delete_ids = [self.id_to_idx[id_] for id_ in ids if id_ in self.id_to_idx]

        for id_ in ids:
            if id_ in self.id_to_idx:
                idx = self.id_to_idx[id_]
                del self.id_to_idx[id_]
                del self.idx_to_id[idx]

        if delete_ids:
            self.index.deleteGTI_by_id(delete_ids)

    def query(self, X, k):
        results = []
        for i in range(X.shape[0]):
            # 确保使用正确的维度
            query_vec = X[i].astype(np.float32).copy()
            if len(query_vec) != self.dim:
                raise ValueError(f"查询向量维度错误: 期望 {self.dim}, 实际 {len(query_vec)}")

            if i%100==0:
                print("[query]i = ", i)

            internal_ids = self.index.query(
                query_vec,
                self.params['L'],
                k
            )
            mapped_ids = [self.idx_to_id.get(i, -1) for i in internal_ids[:k]]
            results.append(mapped_ids)

        self.res = np.array(results)


    def set_query_arguments(self, query_args):
        self.params.update(query_args)

    def index_name(self, name):
        """生成索引文件名"""
        return f"data/{name}.{self.indexkey}"


