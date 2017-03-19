import os,sys
from PIL import Image

# Delete a gif once it's been split into frames
removeProcessedGifs = False

def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0: 
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass

def gen_frames(im, name,compressionSize):
    for i, frame in enumerate(iter_frames(im)):
        x = (name + '%d') % i
        # frame.thumbnail(tuple([compressionSize*x for x in frame.info['size']]), Image.ANTIALIAS)
        frame.thumbnail(tuple(compressionSize*x for x in frame.size), Image.ANTIALIAS)
        # frame.convert("RGBA").save(x+".jpg", optimize=True, quality=100, **frame.info)
        frame.save(x+".png", optimize=True, quality=100, **frame.info)
        frame.close()

def split(split_path,compressionSize):
    for i in range(0, 102068):
        ii = str(i)
        if os.path.isfile(split_path + ii + '.gif'):
            print i
            try:
                os.mkdir(split_path + ii)
                im = Image.open(split_path + ii + '.gif')
                gen_frames(im, split_path + ii + '/' + ii,compressionSize)
            except Exception:
                continue
            im.close()
            if removeProcessedGifs:
                os.remove(split_path + ii + '.gif')
