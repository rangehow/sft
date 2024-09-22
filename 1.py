import pickle
import msgpack
from collections import defaultdict
import random
import timeit

# 生成带有随机数据的 defaultdict(list)
def generate_random_defaultdict(size=1000, list_size=10):
    d = defaultdict(list)
    for i in range(size):
        key = f"key_{i}"
        d[(key)] = [random.randint(0, 1000) for _ in range(list_size)]
    return d

# pickle 序列化和反序列化测试
def pickle_test(data):
    # 使用 protocol 5
    pickled_data = pickle.dumps(data, protocol=5)
    unpickled_data = pickle.loads(pickled_data)

# msgpack 序列化和反序列化测试
def msgpack_test(data):
    # 使用 msgpack 序列化和反序列化
    packed_data = msgpack.packb(data, use_bin_type=True)
    unpacked_data = msgpack.unpackb(packed_data, raw=False)

if __name__ == "__main__":
    # 生成测试数据
    data = generate_random_defaultdict(size=10000, list_size=1000)

    # 测试 pickle 的序列化和反序列化性能
    pickle_time = timeit.timeit(lambda: pickle_test(data), number=100)
    print(f"Pickle (protocol 5) 100次序列化和反序列化时间: {pickle_time:.3f} 秒")

    # 测试 msgpack 的序列化和反序列化性能
    msgpack_time = timeit.timeit(lambda: msgpack_test(data), number=100)
    print(f"Msgpack 100次序列化和反序列化时间: {msgpack_time:.3f} 秒")