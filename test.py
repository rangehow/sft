a=[[[1,2],[3,4]],[[1,2],[3,4]]]
# aaa=[[1,2],[3,4],[1,2],[3,4]]

aaa = [item for sublist in a for item in sublist]
print(aaa)