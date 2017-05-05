import os
import numpy as np
from PIL import Image

file_list=os.listdir("gif_frames")
count=0
for file in file_list:
	array=np.asarray(Image.open("gif_frames/"+file))
	try:
		shape=np.shape(array)[2]
		print np.shape(array)
	except:
		os.remove("gif_frames/"+file) 
	count+=1
assert count==5198
print "Done!"