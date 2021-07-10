import numpy as np
import cv2
import matplotlib.pyplot as plt

def DiffrancesAndThreshold(path,frameCount):
    ''' 
    parameters
        path: string folder path
    extract keyframes from a video
    return: np.array of keyframes 
    '''

    current = cv2.imread(path+"/iframe-1.jpeg")
    frameHeight,frameWidth = current.shape[0],current.shape[1]
    currentHist = np.array([cv2.calcHist([current],[i],None,[256],[0,256]) for i in [0,1,2]]).reshape(256,3)
    currentHist = currentHist / (frameHeight*frameWidth)
    diffHists = np.empty((frameCount,256,3))
    diffHists[0] = currentHist
    for i in range(1,frameCount):
        new = cv2.imread(path+f"/iframe-{i+1}.jpeg")
        newHist = np.array([cv2.calcHist([new],[i],None,[256],[0,256]) for i in [0,1,2]]).reshape(256,3)
        newHist = newHist / (frameHeight*frameWidth)
        diffHists[i] = np.abs(newHist-currentHist)
        currentHist = newHist

    mean = np.mean(diffHists,axis=0)
    return diffHists, mean

def KeyframeExtract(path,frameCount):
    diffHists,threshold = DiffrancesAndThreshold(path,frameCount)
    threshold = np.sum(threshold)
    keyframes = []
    for i,dif in enumerate(diffHists):
        if np.sum(dif) >= threshold:
            keyframes.append(i+1)

    return np.array(keyframes)
