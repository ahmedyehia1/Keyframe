import numpy as np
import cv2
import matplotlib.pyplot as plt

def DiffrancesAndThreshold(path):
    ''' 
    parameters
        path: string folder path
    extract keyframes from a video
    return: np.array of keyframes 
    '''

    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diffHists = np.zeros((frameCount,256,3))
    ret,current = cap.read()
    currentHist = np.array([cv2.calcHist([current],[i],None,[256],[0,256]) for i in [0,1,2]]).reshape(256,3)
    currentHist = currentHist / (frameHeight*frameWidth)
    diffHists[0] = currentHist
    for i in range(1,frameCount):
        ret,new = cap.read()
        newHist = np.array([cv2.calcHist([new],[i],None,[256],[0,256]) for i in [0,1,2]]).reshape(256,3)
        newHist = newHist / (frameHeight*frameWidth)
        diffHists[i] = np.abs(newHist-currentHist)
        currentHist = newHist

    mean = np.mean(diffHists,axis=0)
    std = np.std(diffHists,axis=0)

    cap.release()
    return diffHists, mean+std

def KeyframeExtract(path):
    diffHists,threshold = DiffrancesAndThreshold(path)
    threshold = np.sum(threshold)
    print(threshold)
    keyframes = []
    for i,dif in enumerate(diffHists):
        if np.sum(dif) >= threshold:
            keyframes.append(i)

    return np.array(keyframes)

print(KeyframeExtract("../footage1.mp4"))




def get_files(folder_path):
    only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for i in range(len(only_files)):
        only_files[i] = folder_path + only_files[i]
    return only_files