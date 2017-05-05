import os
import glob

SampleSet=glob.glob("sample_images/4/*.jpg")
SampleSet.sort(key=lambda f: int(filter(str.isdigit, f)))
for i in range(len(SampleSet)):
	os.rename(SampleSet[i],"sample_images/4/1-"+str(i).zfill(3)+".jpg")