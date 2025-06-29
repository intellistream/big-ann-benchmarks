import numpy as np
from gti_module import GTI, Objects

# 1. 构造数据：10 个点，每个维度 3
num_points = 10000
dim = 100
data_array = np.random.rand(num_points, dim).astype(np.float32).copy()

# 2. 设置对象数据
objs = Objects()
objs.setData(data_array)

# Debug 打印确认
print("Data shape:", data_array.shape)

# 3. 初始化 GTI
gti = GTI()

# 4. 构建 GTI 索引
gti.buildGTI(100, 100, 4, objs)

# 6. 查询
query = data_array[0]
results = gti.query(query, 10, 1)
print("Query results:", results)


objs_2 = Objects()
data_array_2 = np.random.rand(50000, dim).astype(np.float32).copy()
objs_2.setData(data_array_2)
gti.insertGTI(objs_2)

objs_1 = Objects()
data_array_1 = np.random.rand(500, dim).astype(np.float32).copy()
objs_1.setData(data_array_1)
gti.insertGTI(objs_1)
# 7. 删除
# del_objs = Objects()
# del_objs.setData(data_array[0:500])
# gti.deleteGTIdeleteGTI_by_id(del_objs)
delete_oids = list(range(0, 1000))

# 调用函数
gti.deleteGTI_by_id(delete_oids)


# 8. 再查
# print(query)
results_after = gti.query(query, 10,1)
# print(len(results_after))
print("Query results after deletion:", results_after)
