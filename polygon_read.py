from __future__ import print_function
import numpy as np
import imageio
import tensorflow as tf

def loadImages(folder,trgFile,n):
    def load_pics(folder,n):
        imgs = []
        for i in range(n):
            img = imageio.imread(folder+"img_{:05}.png".format(i+1))
            ch = img[:,:,0]
            imgs.append(ch)
        return np.array(imgs)

    def load_labels(fn):
        return np.loadtxt(fn, usecols=0)

    pic = load_pics(folder+"/", n)
    ndata, width, height = pic.shape

    inp = (pic/np.float32(255)).reshape(n, width, height, 1)
    trg = load_labels(trgFile)
    trg = trg[0:n]

    return inp, trg, width, height


def loadData(nTrn, nTst):
    # Load data
    (trnInp, trnTrg, imgW, imgH) = loadImages("polyAll-trn", "polyAll-trn_trg.csv", nTrn)
    (tstInp, tstTrg, imgW, imgH) = loadImages("polyAll-tst", "polyAll-tst_trg.csv", nTst)

    if tf.keras.backend.image_data_format() == 'channels_first':
        trnInp = trnInp.reshape(trnInp.shape[0], 1, imgH, imgW)
        tstInp = tstInp.reshape(tstInp.shape[0], 1, imgH, imgW)
        input_shape = (1, imgH, imgW)
    else:
        trnInp = trnInp.reshape(trnInp.shape[0], imgH, imgW, 1)
        tstInp = tstInp.reshape(tstInp.shape[0], imgH, imgW, 1)
        input_shape = (imgH, imgW, 1)

    print('trnInp shape:', trnInp.shape)
    print('tstInp shape:', tstInp.shape)

    trnTrg /= 10;
    tstTrg /= 10;

    return trnInp, trnTrg, tstInp, tstTrg, input_shape

# Get the training data
(trnInp, trnTrg, tstInp, tstTrg, input_shape) = loadData(5000,5000)

