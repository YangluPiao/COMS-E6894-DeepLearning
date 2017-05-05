import os
from PIL import Image
import shutil

path="30000/"

# origin=os.listdir("gif_frames")
# count=0
# for i,file in enumerate(origin):
# 	print i
# 	# print "get file names:"
# 	if count>31:
# 		count=0
# 	gen_file=path+str(i/32)+"_gen_"+str(count)+".jpg"
# 	# print gen_file
# 	hr_file=path+str(i/32)+"_hr_"+str(count)+".jpg"
# 	lr_file=path+str(i/32)+"_lr_"+str(count)+".jpg"
# 	count+=1

# 	frame=file.split('-')[1]
# 	idx=file.split('-')[0]
# 	gen_path='./dataset/'+str(idx)+'/gen'
# 	hr_path='./dataset/'+str(idx)+'/hr'
# 	lr_path='./dataset/'+str(idx)+'/lr'
# 	if not os.path.exists('./dataset/'+str(idx)):
# 		# print "generate folders:"
# 		os.mkdir('./dataset/'+str(idx))
# 		os.mkdir(gen_path)
# 		os.mkdir(hr_path)
# 		os.mkdir(lr_path)

# 	# print "moving: "
# 	os.rename(gen_file, gen_path+'/'+frame.zfill(4))
# 	os.rename(hr_file, hr_path+'/'+frame.zfill(4))
# 	os.rename(lr_file, lr_path+'/'+frame.zfill(4))

# path=os.listdir("./dataset")
# gif_path="../data/gifs/face/"

# for i,file in enumerate(path):
# 	print i
# 	gif=gif_path+file+".gif"
# 	copyfile(gif,"./dataset/"+file+"/"+file+".gif")

path=os.listdir("/home/pyl/Desktop/my-repo/pixel/dataset")
path.sort(key=lambda f: int(filter(str.isdigit, f)))
for i,folder in enumerate(path):
	print i,folder
	# rename_gif="/home/pyl/Desktop/my-repo/pixel/dataset/"+folder+"/"+str(i)+".gif"
	# origin_gif="/home/pyl/Desktop/my-repo/pixel/dataset/"+folder+"/"+folder+".gif"
	# print origin_gif
	# shutil.move(origin_gif,rename_gif)
	rename_folder="dataset/"+str(i)
	origin_folder="dataset/"+folder
	os.rename(origin_folder,rename_folder)