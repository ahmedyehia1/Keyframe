from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sys

def keyframe(name,threshold=0.3):
    cap = cv2.VideoCapture(name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret,current = cap.read()
    prev = current
    keyframes = [cv2.cvtColor(prev,cv2.COLOR_BGR2RGB)]

    prevHist = np.array([cv2.calcHist([current],[0],None,[256],[0,256]),
                         cv2.calcHist([current],[1],None,[256],[0,256]),
                         cv2.calcHist([current],[2],None,[256],[0,256])])

    for dummy in range(1,frameCount):
        if ret:
            ret, current = cap.read()
            currentHist = np.array([cv2.calcHist([current],[0],None,[256],[0,256]),
                                    cv2.calcHist([current],[1],None,[256],[0,256]),
                                    cv2.calcHist([current],[2],None,[256],[0,256])])
            dif = np.sum(np.abs(prevHist-currentHist))/frameWidth/frameHeight/3
            if dif > threshold:
                keyframes.append(cv2.cvtColor(current,cv2.COLOR_BGR2RGB))
            prev = current
            prevHist = currentHist

    cap.release()
    return np.array(keyframes)

def show(keys):
    n = len(keys)
    c = int(np.sqrt(n))
    r = int(np.ceil(n/c))
    fig,a = plt.subplots(r,c,figsize=(20,20))
    fig.suptitle('Keyframes')
    for i in range(n):
        a[int(i/c)][i%c].imshow(keys[i])
        a[int(i/c)][i%c].axis('off')
    for i in range(n,r*c):
        a[int(i/c)][i%c].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("keyframes.png")
    plt.show()

start = time()
keys = keyframe("footage.mp4")
print(f"time taken: {time()-start} seconds")
show(keys)