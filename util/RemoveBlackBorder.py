import sys

from PIL import Image, ImageChops
import os


def trim(im):
	bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox)

src_direcotry = os.path.dirname(sys.argv[1])
dst_direcotry = os.path.dirname(sys.argv[2])

# for root, subdirs, files in os.walk(walk_dir):
#     print('--\nroot = ' + root)
#     list_file_path = os.path.join(root, 'my-directory-list.txt')
#     print('list_file_path = ' + list_file_path)
for filename in os.listdir(sys.argv[1]):
	f = os.path.join(src_direcotry, filename)
	if os.path.isfile(f):
		print("Open:" + f)
		im = Image.open(f)
		image = trim(im)
		image.save("{0}/{1}".format(dst_direcotry, os.path.basename(f)))
		image.close()
