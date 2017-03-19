import imageio
import os
from PIL import Image

# reader = imageio.get_reader("1.gif")
# for i,im in enumerate(reader):
#     imageio.imwrite("image/"+str(i)+".png", im)
def analyseImage(path):
    '''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    '''
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results

def processImage(idx,path,tag):
    '''
    Iterate the GIF, extracting each frame.
    '''
    mode = analyseImage(path)['mode']
    im = Image.open(path)
    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')
    try:
        while True:
            # print "saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile)
            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                im.putpalette(p)
            new_frame = Image.new('RGBA', im.size)
            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)
            new_frame.paste(im, (0,0), im.convert('RGBA'))
            # new_frame.save('./1/%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i), 'PNG')
            new_frame.save('./gifs/%s/%d/%d.png' % (tag,idx,i), 'PNG')
            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass
'''
def png2gif(pngList, gifPath):
    """Uses imageio to convert a set of png images into a gif
    Inputs:
    pngList: array of png filename strings
    gifPath: Path of generated gif file
    """
    images=[]
    print pngList
    for png in pngList:
        images.append(imageio.imread(png))
    imageio.mimsave(gifPath, images)
'''
def png2gif(pngList,gifPath,original_gif,idx):
	original_gif=Image.open(original_gif)
	with imageio.get_writer(gifPath+"/"+str(idx)+'.gif', \
		mode='I',duration=original_gif.info['duration']/1000.0) as writer:
		for png in pngList:
			image = imageio.imread(png)
			writer.append_data(image)
# if __name__ == '__main__':
	# processImage(0,"./0.gif")
	# path="./0/"
	# filenames=os.listdir(path)
	# filenames.sort(key=lambda f: int(filter(str.isdigit, f)))
	# for idx, filename in enumerate(filenames):
	# 	filename=path+filename
	# 	filenames[idx]=filename
	# print filenames
	# png2gif(filenames,"./","./0.gif")