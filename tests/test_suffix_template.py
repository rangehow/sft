import math


class CompressedCounter:
    def __init__(self):
        self.counts = {}  # 只存储非零计数
        self.num_values = 0

    def add(self, value):
        if value in self.counts:
            self.counts[value] += 1
        else:
            self.counts[value] = 1
            self.num_values += 1

    def get(self, value):
        return self.counts.get(value, 0)

    def merge(self, other):
        for value, count in other.counts.items():
            if value in self.counts:
                self.counts[value] += count
            else:
                self.counts[value] = count
                self.num_values += 1


class TrieNode:
    def __init__(self):
        self.children = {}
        self._counts = None  # 懒初始化

    def add_value(self, value):
        if self._counts is None:
            self._counts = CompressedCounter()
        self._counts.add(value)

    def get_counts(self):
        return self._counts


class SuffixTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, input_id:list[int], label:list[int],window_size=math.inf):
        """_summary_

        Args:
            input_id (list[int]): _description_
            label (list[int]): _description_
            window_size (_type_, optional): Defaults to infinite window size.
        """
        node = self.root
        for key,value in zip(reversed(input_id),reversed(label)):
            if key not in node.children:
                node.children[key] = TrieNode()
            node = node.children[key]
        node.add_value(value)

    def search():
        ...

    def merge(self, other):
        def _merge_nodes(node1, node2):
            if node2._counts is not None:
                if node1._counts is None:
                    node1._counts = CompressedCounter()
                node1._counts.merge(node2._counts)
            
            for key, child_node2 in node2.children.items():
                if key in node1.children:
                    _merge_nodes(node1.children[key], child_node2)
                else:
                    node1.children[key] = child_node2

        _merge_nodes(self.root, other.root)

def test_suffix_trie_with_fallback():
    trie = SuffixTrie()
    
    # 插入一些测试数据
    trie.insert(['A', 'B', 'C'], 1)
    trie.insert(['B', 'C'], 2)
    trie.insert(['C'], 3)
    
    print("测试1: 完整匹配")
    query = ['A', 'B', 'C']
    results = trie.search_with_fallback(query)
    print(f"查询 {query}")
    for suffix, counts in results:
        print(f"匹配后缀 {suffix}: {counts.counts}")
    
    print("\n测试2: 需要退避的查询")
    query = ['D', 'A', 'B', 'C']
    results = trie.search_with_fallback(query)
    print(f"查询 {query}")
    for suffix, counts in results:
        print(f"匹配后缀 {suffix}: {counts.counts}")
    
    print("\n测试3: 使用search_first_match")
    result = trie.search_first_match(['D', 'A', 'B', 'C'])
    if result:
        suffix, counts = result
        print(f"第一个匹配: {suffix}, 计数: {counts.counts}")
    
    print("\n测试4: 字符串测试")
    str_trie = SuffixTrie()
    str_trie.insert(list("hello"), 1)
    str_trie.insert(list("ello"), 2)
    str_trie.insert(list("llo"), 3)
    
    query = list("world hello")
    print(f"查询: {''.join(query)}")
    results = str_trie.search_with_fallback(query)
    for suffix, counts in results:
        print(f"匹配后缀 {''.join(suffix)}: {counts.counts}")

if __name__ == "__main__":
    test_suffix_trie_with_fallback()