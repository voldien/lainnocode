import re
import zipfile
from io import StringIO, BytesIO  ## for Python 3

#import PIL.Image
import numpy as np
#import PIL
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from numpy import asarray

import concurrent.futures as cf


path_dataset = "/media/data-sets/mnist-digit-audio.zip"

regex = re.compile('.wav')
with zipfile.ZipFile(path_dataset, 'r') as zip:
	zlist = zip.namelist()
	nr_chunks = 32
	print(list(filter(regex.match, zlist)))

sample_rate, samples = wavfile.read( '/media/data-sets/mnist-digit-audio/data/01/2_01_0.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

def loadDataSet(path=""):
	return (0,0), (0,0)

(train_X, train_Y), (test_X, test_Y) = loadDataSet()




if __name__ == '__main__':
    pass