import os
from glob import glob
from retro.data import merge
from imageio import mimwrite
from cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_RGB2BGR
from datetime import datetime, timedelta

# install roms
def install_roms_in_folder(path):
    for rom in os.listdir(path):
        merge(path + rom, quiet=False)
        
# convert the frame_array to either a GIF or AVI file and save.
def convert_frames(frame_array, directory, fileName, fps=60, otype='AVI'):
    print('Creating replay ...', end=' ')
    if not os.path.exists(directory): os.makedirs(directory)
    if otype == 'AVI':
        fileName += '.avi'
        height, width, layers = frame_array[0].shape
        if layers == 1:
            layers = 0
        size = (width, height)
        fullPath = os.path.join(directory, fileName)
        out = VideoWriter(fullPath, VideoWriter_fourcc(*'DIVX'), fps, size, layers)
        for i in range(len(frame_array)):
            bgr_img = cvtColor(frame_array[i], COLOR_RGB2BGR)
            out.write(bgr_img)
        out.release()
    elif otype == 'GIF':
        fileName += '.gif'
        fullPath = os.path.join(directory, fileName)
        mimwrite(fullPath, frame_array, fps=fps)
    else:
        print('Error: Invalid type, oType must be GIF or AVI.')
        return
    print('Done. Saved to {}'.format(os.path.abspath(directory)))
    
    
# get current date/time foramtted as string
def Now(separate=True):
    now = datetime.now() #- timedelta(hours=6) #timedelata needed for Colab
    if separate:
        return now.strftime('%m_%d_%Y_%H%M%S')
    else:
        return now.strftime('%m%d%Y%H%M%S')
    
# return the most recent file in directory, allows pattern matching
def get_latest_file(directory_matching):
    list_of_files = glob(directory_matching) # * means all if need specific format then *.csv
    if not list_of_files: return None #no matching files found
    return max(list_of_files, key=os.path.getctime)