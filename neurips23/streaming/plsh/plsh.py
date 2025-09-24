import numpy as np
from neurips23.streaming.base import BaseStreamingANN
import plsh_python
import time


class PLSH(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params
        self.name = "plsh"
        self.is_built = False
        self.indexkey = index_params.get("indexkey", "NA")
        self.insert_count = 0  # 记录插入次数
        self.merge_threshold = index_params.get("merge_threshold", 50000)  # 合并阈值


    def setup(self, dtype, max_pts, ndim):
        self.ndim = ndim
        self.max_pts = max_pts

        # 从参数中获取PLSH配置
        k = self.index_params.get("k", 10)  # LSH中的哈希函数数量
        m = self.index_params.get("m", 5)  # LSH表的数量
        num_threads = self.index_params.get("num_threads", 1)

        # 初始化PLSH索引
        self.index = plsh_python.Index(ndim, k, m, num_threads)

    def insert(self, X, ids):
        X = X.astype(np.float32)
        ids = ids.astype(np.uint32)

        if not self.is_built:
            # 第一次插入数据，使用build方法
            self.index.build(X, X.shape[0], ids.tolist())
            self.is_built = True
        else:
            # 后续插入使用insert方法
            self.index.insert(X, ids.tolist())
            self.insert_count += X.shape[0]

            # 根据策略决定是否合并
            if self._should_merge():
                self.index.merge_delta_to_static()

    def _should_merge(self):
        """决定是否进行合并的策略"""
        # 策略1: 基于插入数量阈值
        # 当累积插入的数据量达到阈值时触发合并
        if self.insert_count >= self.merge_threshold:
            print(f"[PLSH] 触发合并: 已插入 {self.insert_count} 个数据点，阈值为 {self.merge_threshold}")
            self.insert_count = 0  # 重置计数器，为下次合并做准备
            return True
        return False

    def delete(self, ids):
        # PLSH暂不支持删除操作
        pass

    def query(self, X, k):
        # 预处理：一次性转换数据类型
        X = X.astype(np.float32)
        n = X.shape[0]
        results = np.zeros((n, k), dtype=np.uint32)
        dists = np.zeros((n, k), dtype=np.float32)

        latencies = []  # 保存每次查询耗时
        n = 50
        start_time = time.time()

        for i in range(n):
            q_start = time.time()

            # 查询
            tags, distances = self.index.query_topk(X[i], k)

            q_end = time.time()
            latencies.append(q_end - q_start)

            actual_k = min(len(tags), k)
            if actual_k > 0:
                results[i, :actual_k] = np.array(tags[:actual_k], dtype=np.uint32) - 1
                dists[i, :actual_k] = distances[:actual_k]

            if actual_k < k:
                if actual_k > 0:
                    results[i, actual_k:] = results[i, actual_k - 1]
                    dists[i, actual_k:] = np.inf
                else:
                    results[i, :] = 0
                    dists[i, :] = np.inf

        end_time = time.time()
        total_time = end_time - start_time

        # QPS
        qps = n / total_time

        # latency
        latencies_ms = np.array(latencies) * 1000
        avg_latency = np.mean(latencies_ms)
        p95_latency = np.percentile(latencies_ms, 95)
        p99_latency = np.percentile(latencies_ms, 99)

        print(f"[Query Stats] Total Queries: {n}")
        print(f"[Query Stats] Total Time: {total_time:.4f} sec")
        print(f"[Query Stats] QPS: {qps:.2f}")
        print(f"[Query Stats] Avg Latency: {avg_latency:.3f} ms/query")
        print(f"[Query Stats] P95 Latency: {p95_latency:.3f} ms/query")
        print(f"[Query Stats] P99 Latency: {p99_latency:.3f} ms/query")

        self.res = results
        return results, dists

    def set_query_arguments(self, query_args):
        # PLSH的查询参数（如果需要可以扩展）
        pass

    def index_name(self, name):
        return f"data/{name}.plshindex"

    def merge_delta_to_static(self):
        """合并增量数据到静态索引"""
        if hasattr(self.index, 'merge_delta_to_static'):
            print("Mergeing delta to static")
            self.index.merge_delta_to_static()