from subprocess import call
import glob
import os
import shutil
from PIL import Image


tag="cat"
recovered_path="recovered_gifs_"+tag

split_path="./gifs/"+tag+"/"
gif_path=glob.glob(split_path+"*.gif")

gif_path.sort(key=lambda f: int(filter(str.isdigit, f)))

def gif2images():
	for i in range(len(gif_path)):
		if not os.path.exists(split_path+"/"+str(i)):
			os.mkdir(split_path+"/"+str(i))
	for idx, gif in enumerate(gif_path):
		print("Splitting %d.gif with tag '%s'"%(idx,tag))
		# May need to modify the path.
		call(["convert",gif_path[idx],"gifs/"+tag+"/"+str(idx)+"/%05d.png"])
	print "Done with splitting." 
def images2gif():
	if os.path.exists(recovered_path):
		boo = raw_input("recovered gifs already exist, overwrite?y/n")
		if boo is 'y':
			shutil.rmtree(recovered_path)
			os.mkdir(recovered_path)
	else:
		os.mkdir(recovered_path)
	for i in range(len(gif_path)):
		print("Recovering %d.gif"%(i))
		png_path=split_path+str(i)+"/"
		pngList=glob.glob(png_path+"*.png")
		frames = len(pngList)
		original_gif=Image.open(gif_path[i])
		delay = original_gif.info['duration']/10.0
		# May need to modify the paths.
		call(["convert","-delay",str(delay),"-loop","0",png_path+"/000*.png",recovered_path+"/"+str(i)+".gif"])
	print "Done with recovering." 
if __name__ == '__main__':
	gif2images()
	images2gif()