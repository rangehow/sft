# test_statistic_consistency.py

from collections import Counter
import multiprocessing

from tqdm import tqdm

# Define the Args class
class Args:
    def __init__(self, clm=True, ngram=0, mono=False):
        self.clm = clm
        self.ngram = ngram
        self.mono = mono

# Define the TrieNode and Trie classes
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
        def _merge_nodes(node1, node2):
            node1.value.update(node2.value)
            for key, child_node2 in node2.children.items():
                if key in node1.children:
                    _merge_nodes(node1.children[key], child_node2)
                else:
                    node1.children[key] = child_node2
        _merge_nodes(self.root, other.root)
        
    def __str__(self):
        def _print(node, prefix):
            if node.value:
                print(f"{prefix}: {dict(node.value)}")
            for key, child in node.children.items():
                _print(child, prefix + [key])
        _print(self.root, [])
        
    # Helper function to convert the Trie to a dict for comparison
    def to_dict(self):
        result = {}
        def _collect(node, prefix):
            if node.value:
                result[tuple(prefix)] = dict(node.value)
            for key, child in node.children.items():
                _collect(child, prefix + [key])
        _collect(self.root, [])
        return result

# The process_chunk_statistic and process_mono_chunk functions need to be at the top level
def process_chunk_statistic(chunk_data, args_ngram, args_clm):
    local_supervised_trie = Trie()
    local_clm_trie = Trie() if args_clm else None
    local_max_target_len = 0

    for data_point in chunk_data:
        input_id, label = data_point["input_ids"], data_point["labels"]
        assert len(input_id) == len(label)
        if args_clm:
            flag4LossArea = False
            regionBeginIdx = -1

        local_max_target_len = max(local_max_target_len, len(label))
        for i in range(len(label) - 1):
            if label[i + 1] != -100:
                local_supervised_trie.insert(input_id[: i + 1], label[i + 1])

                if args_clm:
                    if flag4LossArea is False:
                        regionBeginIdx = i
                        flag4LossArea = True

                    if i - regionBeginIdx >= args_ngram:
                        clm_key = label[regionBeginIdx + 1 : i + 1]
                        local_clm_trie.insert(clm_key, label[i + 1])
            elif args_clm and flag4LossArea:
                flag4LossArea = False

    return local_supervised_trie, local_clm_trie, local_max_target_len

def process_mono_chunk(chunk_data, args_ngram, max_target_len):
    local_clm_trie = Trie()
    for data_point in chunk_data:
        input_id, label = data_point["input_ids"], data_point["labels"]
        index_for_start_area = 0
        for i in range(len(label) - 1):
            if label[i] == -100:
                index_for_start_area += 1
            if (
                label[i + 1] != -100
                and i - index_for_start_area >= args_ngram - 1
                and i - index_for_start_area <= max_target_len + 5
            ):
                clm_key = label[index_for_start_area : i + 1]
                local_clm_trie.insert(clm_key, label[i + 1])
    return local_clm_trie



def statistic(args,train_dataset,mono_dataset=None):

        supervised_trie = Trie()
        # supervised_dict = defaultdict(Counter)
        if args.clm:
            clm_trie = Trie()
            # clm_dict = defaultdict(Counter)   
        
        max_target_len=0 # 用于减少mono阶段无意义的统计。


        # 统计begin-----------------------------------------------------------------------------------
        for j in tqdm(range(len(train_dataset)), desc="statistic stage"):

            input_id, label = train_dataset[j]["input_ids"], train_dataset[j]["labels"]
            assert len(input_id) == len(label)
            # 如果是多轮对话,那么label将是穿插着多段连续-100的int list
            # supervised信号的key是第一段-100结束开始,以后开始递增
            # clm信号的key应该是非-100区域内独立统计
            if args.clm:
                # 用于标志是否到达非-100区域的,这里有个假定就是开头一定是连续的-100区域[通常因为开头是特殊标记,所以总是的]
                # 这个标记主要的作用就是为了辅助regionBeginIdx更新
                flag4LossArea = False
                # 用于辅助ngram测算现在是否可以更新clm了
                regionBeginIdx = -1

            max_target_len=max(max_target_len,len(label))
            for i in range(len(label) - 1):

                if label[i + 1] != -100:

                    # supervised_key = tuple(input_id[: i + 1])
                    # supervised_dict[supervised_key].update([label[i + 1]])
                    supervised_trie.insert(input_id[: i + 1], label[i + 1])

                    if args.clm:
                        if flag4LossArea is False:
                            # 此时下一个label不是-100，但是regionBeginIdx本身指向的还是-100
                            regionBeginIdx = i
                            flag4LossArea = True

                        if i - regionBeginIdx >= args.ngram:
                            # clm_key = tuple(label[regionBeginIdx + 1 : i + 1])
                            # clm_dict[clm_key].update([label[i + 1]])
                            clm_trie.insert(
                                label[regionBeginIdx + 1 : i + 1], label[i + 1]
                            )
                elif args.clm and flag4LossArea:
                    flag4LossArea = False

        if args.mono and args.clm:

            for j in tqdm(range(len(mono_dataset)), desc="mono statistic stage"):
                input_id, label = (
                    mono_dataset[j]["input_ids"],
                    mono_dataset[j]["labels"],
                )
                index_for_start_area=0
                for i in range(len(label) - 1):
                    if label[i]==-100:
                        index_for_start_area+=1
                    if label[i + 1] != -100 and i-index_for_start_area>=args.ngram-1 and i-index_for_start_area<= max_target_len+5: # +5没什么实际意义只是随便设置的，怕不准确
                        # import pdb
                        # pdb.set_trace()
                        clm_trie.insert(label[index_for_start_area: i + 1], label[i + 1])
                        

        return supervised_trie, clm_trie



# Define the modified statistic function (with multiprocessing)
def statistic_modified(args, train_dataset, mono_dataset=None):
    import math

    supervised_trie = Trie()
    clm_trie = Trie() if args.clm else None
    max_target_len = 0

    num_workers = multiprocessing.cpu_count()
    num_workers = max(1, min(num_workers - 1, len(train_dataset)))

    chunk_size = math.ceil(len(train_dataset) / num_workers)
    chunks = [
        train_dataset[i:i+chunk_size]
        for i in range(0, len(train_dataset), chunk_size)
    ]

    pool_args = [(chunk, args.ngram, args.clm) for chunk in chunks]

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(process_chunk_statistic, pool_args)

    for local_supervised_trie, local_clm_trie, local_max_target_len in results:
        supervised_trie.merge(local_supervised_trie)
        if args.clm and local_clm_trie:
            clm_trie.merge(local_clm_trie)
        max_target_len = max(max_target_len, local_max_target_len)

    if args.mono and args.clm:
        chunk_size = math.ceil(len(mono_dataset) / num_workers)
        mono_chunks = [
            mono_dataset[i:i+chunk_size]
            for i in range(0, len(mono_dataset), chunk_size)
        ]

        pool_args = [(chunk, args.ngram, max_target_len) for chunk in mono_chunks]

        with multiprocessing.Pool(num_workers) as pool:
            mono_results = pool.starmap(process_mono_chunk, pool_args)

        for local_clm_trie in mono_results:
            clm_trie.merge(local_clm_trie)
    else:
        clm_trie = clm_trie if args.clm else None

    return supervised_trie, clm_trie





# Now write the test code
if __name__ == '__main__':
    # Create minimal args
    args = Args(clm=True, ngram=0, mono=False)

    # Create minimal train_dataset
    train_dataset = [
        {'input_ids': [1, 2, 3, 4], 'labels': [-100, 2, -100, 5]},
        {'input_ids': [6, 7, 8, 9], 'labels': [6, -100, 8, -100]},
        {'input_ids': [10, 11, 12, 13], 'labels': [-100, -100, 12, 14]},
    ]

    # Create minimal mono_dataset if needed
    mono_dataset = [
        {'input_ids': [20, 21, 22], 'labels': [20, 21, -100]},
    ] if args.mono else None

    # Run statistic_original'
    supervised_trie_orig, clm_trie_orig = statistic(args, train_dataset, mono_dataset)

    # Run statistic_modified
    supervised_trie_mod, clm_trie_mod = statistic_modified(args, train_dataset, mono_dataset)
    
    # Compare supervised_trie
    if supervised_trie_orig.to_dict() == supervised_trie_mod.to_dict():
        print("supervised_trie results are consistent.")
    else:
        print("supervised_trie results are NOT consistent!")

    # Compare clm_trie
    if args.clm:
        if clm_trie_orig.to_dict() == clm_trie_mod.to_dict():
            print("clm_trie results are consistent.")
        else:
            print("clm_trie results are NOT consistent!")