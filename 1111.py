from typing import List

def quick_sort(arr: List[int], start: int = None, end: int = None) -> List[int]:
    if start is None:
        start = 0
    if end is None:
        end = len(arr) - 1
        
    if start >= end:
        return arr
    
    # 选择中间位置的值作为枢纽
    pivot_idx = (start + end) // 2
    pivot = arr[pivot_idx]
    
    # 双指针分区
    left, right = start, end
    while left <= right:
        while left <= right and arr[left] < pivot:
            left += 1
        while left <= right and arr[right] > pivot:
            right -= 1
        if left <= right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
    
    # 递归排序左右两部分
    quick_sort(arr, start, right)
    quick_sort(arr, left, end)
    
    return arr

# 测试代码
a = [1, 3, 4, 2, 1]
result = quick_sort(a)
print(result)