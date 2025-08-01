import numpy as np
import dynagraph
from typing import Any, Dict, Optional
import logging

class DynaGraph:
    def __init__(self, metric: str, index_params: Dict[str, Any]):
        self._metric = metric
        self._index_params = index_params
        self._index = None
        self._built = False
        
        # DynaGraph参数
        self._alpha = index_params.get("alpha", 1.2)
        self._coef_L = index_params.get("coef_L", 100)
        self._coef_R = index_params.get("coef_R", 64)
        self._batch_size = index_params.get("batch_size", 2000)
        
        logging.info(f"DynaGraph initialized with parameters: alpha={self._alpha}, "
                    f"coef_L={self._coef_L}, coef_R={self._coef_R}, batch_size={self._batch_size}")
    
    def fit(self, X: np.ndarray) -> None:
        """Build the index with the given data."""
        if self._metric != "euclidean":
            raise ValueError(f"DynaGraph only supports euclidean metric, got {self._metric}")
        
        n_points, dim = X.shape
        logging.info(f"Building DynaGraph index with {n_points} points, {dim} dimensions")
        
        # 创建DynaGraph索引
        self._index = dynagraph.Index()
        
        # 设置参数
        self._index.setup(
            max_points=n_points * 2,  # 留一些余量
            dim=dim,
            alpha=self._alpha,
            coef_L=self._coef_L,
            coef_R=self._coef_R,
            batch_size=self._batch_size
        )
        
        # 创建标签（使用连续的整数）
        tags = np.arange(n_points, dtype=np.uint64)
        
        # 构建索引
        X_contiguous = np.ascontiguousarray(X.astype(np.float32))
        self._index.build(X_contiguous, n_points, tags.tolist())
        
        self._built = True
        logging.info("DynaGraph index built successfully")
    
    def query(self, q: np.ndarray, k: int) -> np.ndarray:
        """Query the index for k nearest neighbors."""
        if not self._built:
            raise RuntimeError("Index must be built before querying")
        
        if q.ndim == 1:
            # 单个查询
            q_contiguous = np.ascontiguousarray(q.astype(np.float32))
            tags, distances = self._index.query(q_contiguous, k)
            return np.array(tags, dtype=np.int32)
        else:
            # 批量查询
            q_contiguous = np.ascontiguousarray(q.astype(np.float32))
            tags, distances = self._index.batch_query(q_contiguous, k, num_threads=8)
            return np.array(tags, dtype=np.int32)
    
    def batch_query(self, Q: np.ndarray, k: int) -> np.ndarray:
        """Batch query the index."""
        return self.query(Q, k)
    
    def __str__(self) -> str:
        return f"DynaGraph(alpha={self._alpha}, coef_L={self._coef_L}, coef_R={self._coef_R}, batch_size={self._batch_size})"


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
    except ImportError:
        return 0.0


# 为了兼容benchmark框架的要求
def instantiate_from_dict(args: Dict[str, Any]) -> DynaGraph:
    """Create DynaGraph instance from parameter dictionary."""
    metric = args.get("metric", "euclidean")
    return DynaGraph(metric, args)
