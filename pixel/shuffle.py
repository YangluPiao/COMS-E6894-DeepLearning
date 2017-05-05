import os
from random import shuffle

file_names=os.listdir("celebA")
file_names.sort(key=lambda f: int(filter(str.isdigit, f)))
sh=file_names[29:]
shuffle(sh)
file_names[29:]=sh
f = open("./data/train.txt",'w')
for file in file_names:
	f.write("/home/pyl/Desktop/COMS-E6894-DeepLearning/pixel/data/celebA/"+file + '\n')
f.close()

