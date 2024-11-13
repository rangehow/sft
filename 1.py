def a(aaa):
    return aaa

try:
    a(1,2)
except TypeError as e:
    print(e)  # 将输出: a() takes 1 positional argument but 2 were given