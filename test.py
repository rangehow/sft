batch=[{
  'supervised':  [[1,2],[3,4]]
},{
  'supervised':  [[1,2],[3,4]]
}]

# aaa=[[1,2],[3,4],[1,2],[3,4]]

aaa =[item for d in batch for item in d['supervised'] ]
print(aaa)