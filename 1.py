import pickle
import lz4.frame
from collections import defaultdict
import random
import timeit
import os

def generate_random_defaultdict(size=1000, list_size=10):
    d = defaultdict(list)
    for i in range(size):
        key = f"key_{i}"
        d[(key)] = [random.randint(0, 1000) for _ in range(list_size)]
    return d

# 准备测试文件
def prepare_test_files(data):
    # 写入普通pickle文件
    with open("normal_data.pkl", 'wb') as f:
        pickle.dump(data, f, protocol=5)
    
    # 写入LZ4压缩文件
    with lz4.frame.open("compressed_data.pkl.lz4", 'wb') as f:
        pickle.dump(data, f, protocol=5)

# 测试普通文件读取
def read_normal_file():
    with open("normal_data.pkl", 'rb') as f:
        loaded_data = pickle.load(f)

# 测试压缩文件读取
def read_compressed_file():
    with lz4.frame.open("compressed_data.pkl.lz4", 'rb') as f:
        loaded_data = pickle.load(f)

if __name__ == "__main__":
    # 生成测试数据并准备文件
    data = generate_random_defaultdict(size=10000, list_size=1000)
    prepare_test_files(data)
    
    # 获取文件大小
    normal_size = os.path.getsize("normal_data.pkl") / (1024 * 1024)  # MB
    compressed_size = os.path.getsize("compressed_data.pkl.lz4") / (1024 * 1024)  # MB
    
    # 测试读取速度
    normal_read_time = timeit.timeit(read_normal_file, number=50)
    compressed_read_time = timeit.timeit(read_compressed_file, number=50)
    
    # 输出结果
    print(f"普通文件大小: {normal_size:.2f} MB")
    print(f"压缩文件大小: {compressed_size:.2f} MB")
    print(f"压缩比: {compressed_size/normal_size:.2%}")
    print(f"普通文件50次读取时间: {normal_read_time:.3f} 秒")
    print(f"压缩文件50次读取时间: {compressed_read_time:.3f} 秒")
    print(f"每次读取时间 - 普通文件: {(normal_read_time/50)*1000:.2f} ms")
    print(f"每次读取时间 - 压缩文件: {(compressed_read_time/50)*1000:.2f} ms")
    
    # 清理测试文件
    os.remove("normal_data.pkl")
    os.remove("compressed_data.pkl.lz4")