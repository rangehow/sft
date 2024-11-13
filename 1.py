from collections import Counter
from typing import List, Dict, Union, Tuple, Optional
import sys
import random
import string
import time
import tracemalloc

# 原始版本
class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = Counter()

class OriginalTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, key_list, value):
        node = self.root
        for key in key_list:
            if key not in node.children:
                node.children[key] = TrieNode()
            node = node.children[key]
        node.value[value] += 1

    def search(self, key_list):
        node = self.root
        for key in key_list:
            if key not in node.children:
                return None
            node = node.children[key]
        return node.value

    def merge(self, other):
        def _merge_nodes(node1, node2):
            node1.value.update(node2.value)
            for key, child_node2 in node2.children.items():
                if key in node1.children:
                    _merge_nodes(node1.children[key], child_node2)
                else:
                    node1.children[key] = child_node2
        _merge_nodes(self.root, other.root)

# 优化版本
class OptimizedTrieNode:
    __slots__ = ['_children', '_value']
    
    def __init__(self):
        self._children = None
        self._value = None
    
    @property
    def children(self) -> Dict:
        if self._children is None:
            self._children = {}
        return self._children
    
    def add_value(self, value: str) -> None:
        if self._value is None:
            self._value = (value, 1)
        elif isinstance(self._value, tuple):
            key, count = self._value
            if key == value:
                self._value = (key, count + 1)
            else:
                counter = Counter()
                counter[key] = count
                counter[value] = 1
                self._value = counter
        else:
            self._value[value] += 1
    
    def get_value(self) -> Optional[Union[Counter, Tuple[str, int]]]:
        return self._value
    
    def get_value_as_counter(self) -> Counter:
        if self._value is None:
            return Counter()
        if isinstance(self._value, tuple):
            key, count = self._value
            return Counter({key: count})
        return self._value

class OptimizedTrie:
    def __init__(self):
        self.root = OptimizedTrieNode()
        self._size = 0
    
    def insert(self, key_list: List[str], value: str) -> None:
        node = self.root
        for key in key_list:
            if node._children is None:
                node._children = {}
            if key not in node.children:
                node.children[key] = OptimizedTrieNode()
                self._size += 1
            node = node.children[key]
        node.add_value(value)
    
    def search(self, key_list: List[str]) -> Optional[Counter]:
        node = self.root
        for key in key_list:
            if node._children is None or key not in node.children:
                return None
            node = node.children[key]
        return node.get_value_as_counter() if node._value is not None else None

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_test_data(num_entries, path_length, value_length):
    data = []
    for _ in range(num_entries):
        path = [generate_random_string(3) for _ in range(path_length)]
        value = generate_random_string(value_length)
        data.append((path, value))
    return data

def measure_memory_and_time(test_data, trie_class, name):
    # 启动内存跟踪
    tracemalloc.start()
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建和填充Trie
    trie = trie_class()
    for path, value in test_data:
        trie.insert(path, value)
    
    # 测试搜索
    for path, _ in test_data[:100]:  # 测试前100个路径
        trie.search(path)
    
    # 记录结束时间
    end_time = time.time()
    
    # 获取内存使用情况
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'name': name,
        'current_memory': current / 1024 / 1024,  # MB
        'peak_memory': peak / 1024 / 1024,        # MB
        'time': end_time - start_time
    }

def run_comparison():
    # 测试参数
    test_cases = [
        {'num_entries': 1000, 'path_length': 3, 'value_length': 5},
        {'num_entries': 10000, 'path_length': 5, 'value_length': 5},
        {'num_entries': 100000, 'path_length': 3, 'value_length': 5},
    ]
    
    for case in test_cases:
        print(f"\nTest case: {case}")
        test_data = generate_test_data(
            case['num_entries'], 
            case['path_length'], 
            case['value_length']
        )
        
        # 测试原始版本
        original_results = measure_memory_and_time(test_data, OriginalTrie, "Original Trie")
        
        # 测试优化版本
        optimized_results = measure_memory_and_time(test_data, OptimizedTrie, "Optimized Trie")
        
        # 打印结果
        print("\nResults:")
        for results in [original_results, optimized_results]:
            print(f"\n{results['name']}:")
            print(f"Current Memory: {results['current_memory']:.2f} MB")
            print(f"Peak Memory: {results['peak_memory']:.2f} MB")
            print(f"Time: {results['time']:.2f} seconds")
        
        # 计算改进
        memory_improvement = ((original_results['current_memory'] - 
                             optimized_results['current_memory']) / 
                            original_results['current_memory'] * 100)
        time_improvement = ((original_results['time'] - 
                           optimized_results['time']) / 
                          original_results['time'] * 100)
        
        print(f"\nImprovements:")
        print(f"Memory: {memory_improvement:.1f}%")
        print(f"Time: {time_improvement:.1f}%")

if __name__ == "__main__":
    run_comparison()