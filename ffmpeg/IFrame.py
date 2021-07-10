import subprocess
from timeit import Timer
import concurrent.futures
from threshold import KeyframeExtract

def Iframes(infile):
    folder = '.'.join(infile.split('.')[:-1])+"_keyframes"
    try:
        subprocess.os.mkdir(folder)
    except: FileExistsError

    ffmpeg = "ffmpeg.exe"

    files = folder + "/iframe-%0d.jpeg"
    subprocess.check_call([ffmpeg, "-skip_frame", "nokey", "-i", f"{infile}", "-vsync", "0" , "-r","30","-f", "image2", f"{files}"])# ,stderr=subprocess.PIPE)
    frameCount = len([name for name in subprocess.os.listdir(folder)])
    
    return folder, frameCount


videoPath = "" # Video File Path
path,frameCount = Iframes(videoPath)
print(KeyframeExtract(path,frameCount))