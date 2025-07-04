import keras
import os
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI

import cv2 as cv
import random
import string
model=keras.models.load_model('ganmodel.h5',compile=False)
app=FastAPI()


def saveimages(images):
    hashlist=[]
    foldername= random.getrandbits(28)
    os.mkdir(str(foldername))
    for a in range(32):
        hash = random.getrandbits(28)
        hash=str(hash)
        I=(images[a]* 255).round().astype(np.uint8)
        cv.imwrite(f"{foldername}/{hash}.png", I)
        plt.imshow(I, cmap='viridis')
        plt.axis('off')
        plt.savefig(f"{foldername}/{hash}.png")

        hashlist.append(f'{hash}.jpg')
    return str(foldername), hashlist


@app.post('/')
async def get_model():
    imagesdata=model.predict(np.random.randn(32, 576))

    fname,hList=saveimages(imagesdata)
    return{
        'foldername':fname,
        'imagename':hList

    }
