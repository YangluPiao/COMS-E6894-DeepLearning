from subprocess import call
import glob
import os
import shutil
import argparse
from PIL import Image


def initConversion(tag):
	"""
	Creates some paths that are shared between defs

	tag - folder name to look in for gifs
	"""

	#tag="dog2"
	recovered_path="recovered_gifs_"+tag

	split_path="./gifs/"+tag+"/"
	gif_path=glob.glob(split_path+"*.gif")
	gif_path.sort(key=lambda f: int(filter(str.isdigit, f)))

	return (recovered_path, split_path, gif_path)

def parse_args():
	"""
	Handles input of tag/folder name on command line
	"""
	parser = argparse.ArgumentParser(description="Generate pngs-gifs")
	parser.add_argument('tag', nargs='+', help="Tags of GIFs to convert to PNG")
	return parser.parse_args()

def gif2images(tag,split_path,gif_path):
	"""
	Converts a gif to pngs with each png containing the full image data

	tag - folder name where files reside
	split_path - path to folder where gifs are stored
	gif_path - path of all gifs in split_path
	"""
	for i in range(len(gif_path)):
		if not os.path.exists(split_path+"/"+str(i)):
			os.mkdir(split_path+"/"+str(i))
	for idx, gif in enumerate(gif_path):
		print("Splitting %d.gif with tag '%s'"%(idx,tag))
		# May need to modify the path.
		call(["convert","-coalesce",gif_path[idx],"gifs/"+tag+"/"+str(idx)+"/%05d.png"])
	print "Done with splitting."

def images2gif(tag,recovered_path,gif_path):
	"""
	Converts a png frames to gif with each png containing the full image data

	tag - folder name where files reside
	recovered_path - path to folder where generated gifs are stored
	gif_path - path of all gifs in split_path
	"""

	if os.path.exists(recovered_path):
		boo = raw_input("recovered gifs already exist, overwrite? : y/n -> ")
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
	args = parse_args()
	tag = args.tag[0]
	recovered_path, split_path, gif_path = initConversion(tag)
	gif2images(tag,split_path,gif_path)
	images2gif(tag,recovered_path,gif_path)
