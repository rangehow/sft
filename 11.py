import timeit

def method1(numbers):
    return next(filter(lambda x: x != -100, numbers), None)

def method2(numbers):
    return next((x for x in numbers if x != -100), None)

def method3(numbers):
    for n in numbers:
        if n != -100:
            return n
    return None

# 测试数据
test_list = [-100] * 10000 + [5] + [-100] * 10000

# 性能测试
t1 = timeit.timeit(lambda: method1(test_list), number=1)
t2 = timeit.timeit(lambda: method2(test_list), number=1)
t3 = timeit.timeit(lambda: method3(test_list), number=1)

print(f"filter方法: {t1:.4f}秒")
print(f"生成器方法: {t2:.4f}秒")
print(f"简单循环: {t3:.4f}秒")


