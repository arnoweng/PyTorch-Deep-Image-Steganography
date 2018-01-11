import os
import random

dir="/scratch/wxy/ImageNet/dataset/"

fileList=[]
for root, dirs, files in os.walk(dir):
    for file in files:
        fileList.append(file)


print(len(fileList))
slice = random.sample(fileList, 5000)
print(len(slice))
txtName = "filelist.txt"
f=open(txtName, "w")
for file in slice:
    f.write(file+"\n")

f.close()