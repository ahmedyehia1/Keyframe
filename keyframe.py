import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture('footage4.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret,current = cap.read()
frameDiff = np.zeros((frameCount,1))
keyframes = [cv2.cvtColor(current,cv2.COLOR_BGR2RGB)]

currentHist = cv2.calcHist([cv2.cvtColor(current,cv2.COLOR_RGB2GRAY)],[0],None,[256],[0,256])


for fc in range(1,frameCount):
    if ret:
        ret, current = cap.read()
        currentHist = cv2.calcHist([cv2.cvtColor(current,cv2.COLOR_RGB2GRAY)],[0],None,[256],[0,256])
        frameHist = cv2.calcHist([cv2.cvtColor(keyframes[-1],cv2.COLOR_RGB2GRAY)],[0],None,[256],[0,256])
        dif = np.sum(np.abs(frameHist-currentHist))/frameWidth/frameHeight/3
        print(fc,dif)
        if dif > 0.3:
            keyframes.append(cv2.cvtColor(current,cv2.COLOR_BGR2RGB))
            fig, axs = plt.subplots(1,2)
            fig.suptitle(f"{dif} error in frame {fc}")
            axs[0].imshow(keyframes[-2])
            axs[1].imshow(keyframes[-1])
            plt.show()
        frameDiff[fc] = dif

cap.release()

print(len(keyframes))
plt.plot(frameDiff)
plt.show()