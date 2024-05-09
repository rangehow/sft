from collections import Counter
import pickle
class TrieNode:
    def __init__(self):
        self.children = {}
        self.c=Counter()


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def update(self, keys:list[int],values:list[int]):
        node = self.root
        # 先保证keys是被创建完了的
        for num in keys:
            if num not in node.children:
                node.children[num] = TrieNode()
            node = node.children[num]
        
        for num in values:
            node.c.update([num])
            if num not in node.children:
                node.children[num] = TrieNode()
            node = node.children[num]    
        node.c.update([values[-1]])


    def search(self, nums):
        node = self.root
        for num in nums:
            if num not in node.children:
                return False
            node = node.children[num]
        return node.c
    
    def serialize(self):
        def serialize_node(node):
            serialized = {
                'children': {},
                'c': dict(node.c)
            }
            for key, child_node in node.children.items():
                serialized['children'][key] = serialize_node(child_node)
            return serialized

        return pickle.dumps(serialize_node(self.root))

    @classmethod
    def deserialize(cls, serialized_data):
        def deserialize_node(data):
            node = TrieNode()
            node.c = Counter(data['c'])
            for key, child_data in data['children'].items():
                node.children[key] = deserialize_node(child_data)
            return node

        trie = cls()
        trie.root = deserialize_node(pickle.loads(serialized_data))
        return trie
    
    def __str__(self):
        return f"Node(children={self.children.keys()}, is_end_of_word={self.is_end_of_word})"