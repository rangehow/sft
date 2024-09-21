import pickle
from collections import Counter

# 定义 TrieNode 和 Trie 类
class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = Counter()

class Trie:
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
        self._merge_nodes(self.root, other.root)

    def _merge_nodes(self, node1, node2):
        # 合并 values
        node1.value.update(node2.value)
        # 合并 children
        for key, child_node2 in node2.children.items():
            if key in node1.children:
                self._merge_nodes(node1.children[key], child_node2)
            else:
                node1.children[key] = child_node2

# 创建并插入数据到 Trie
trie = Trie()
trie.insert(['a', 'b', 'c'], 'value1')
trie.insert(['a', 'b', 'd'], 'value2')

# 检查插入是否成功
print("Before pickling:")
print(trie.search(['a', 'b', 'c']))  # 应该输出 Counter({'value1': 1})
print(trie.search(['a', 'b', 'd']))  # 应该输出 Counter({'value2': 1})

# 尝试使用 pickle 序列化和反序列化
try:
    serialized_trie = pickle.dumps(trie)  # 序列化
    deserialized_trie = pickle.loads(serialized_trie)  # 反序列化
    
    print("After pickling:")
    print(deserialized_trie.search(['a', 'b', 'c']))  # 应该输出 Counter({'value1': 1})
    print(deserialized_trie.search(['a', 'b', 'd']))  # 应该输出 Counter({'value2': 1})

except pickle.PicklingError as e:
    print(f"PicklingError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")