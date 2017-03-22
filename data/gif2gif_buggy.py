# from split import split
from PrepareImageData import *
import os
import shutil
import glob

compressSize=1
tag="women"
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
        processImage(idx,gif_path[idx],tag)
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
        pngList.sort(key=lambda f: int(filter(str.isdigit, f)))
        original_gif=gif_path[i]
        png2gif(pngList,recovered_path,original_gif,i)
    print "Donw with merging." 
if __name__ == '__main__':
    # gif2images()
    images2gif()
    
    