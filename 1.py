import cupy as cp
from scipy.optimize import root

# 定义函数
def f(x):
    return x**2 - 4

# 使用 CuPy 数组
x0 = cp.array([1.0])  # 初始猜测

# 将 CuPy 数组转换为 NumPy 数组进行求解
sol = root(lambda x: cp.asnumpy(f(x)), cp.asnumpy(x0))

# 将结果转换回 CuPy 数组
root_result = cp.array(sol.x)

print("Root:", root_result)
