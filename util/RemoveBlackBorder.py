import sys
from pathlib import Path

from PIL import Image, ImageChops
import os


def trim(im):
	bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox)


src_directory = os.path.dirname(sys.argv[1])
dst_directory = os.path.dirname(sys.argv[2])

for root, subdirs, files in os.walk(src_directory):

	for filename in files:

		open_file_path = os.path.join(root, filename)
		subDirectory = str.removeprefix(open_file_path, src_directory).removeprefix('/')
		subDirectory = os.path.split(subDirectory)[0]

		print('\t- load file %s (full path: %s)' % (filename, open_file_path))
		if os.path.isfile(open_file_path):
			save_file_path = os.path.join(os.path.join(dst_directory, subDirectory), filename)
			Path(os.path.split(save_file_path)[0]).mkdir(parents=True, exist_ok=True)

			try:
				im = Image.open(open_file_path)
				#im.verify()

				image = trim(im)

				print('\t- save file %s (full path: %s)' % (filename, save_file_path))
				image.save(save_file_path)
				image.close()
			except:
				print('\t- Failed to parse file %s (full path: %s)' % (filename, open_file_path))
				pass
