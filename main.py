import os
import shutil
import subprocess

from LoadModel import *
from ModelOutptProcessing import *
from PredImage import *
from PredVid import *
from ShowImage import *

def clean_frames_folder():
    '''
    this function will delete the images in the frames directory
    '''
    folder = 'frames/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    pass

if __name__ == '__main__':

    '''

    model = load_model('config.cfg', 'w.weights')
    # -------------------- single image ----------------------
    # detect cars within image
    w, h, output = predict_image('car.jpg', model)

    # the output of the model is
    # a lot of boxes so we have
    # to filter them and get the
    # best predictions
    boxes, idxs = get_boxes(output, w, h)

    # draw the boxes and show the image
    show_image('car.jpg', boxes, idxs, show=True)
    '''

    # -------------------- video ---------------------
    os.mkdir('frames')  # to save frames extracted from video
    os.mkdir("stream")  # to save video made from frames with boxes

    # load model using the files w.weights and config.cfg
    model = load_model('config.cfg', 'w.weights')
    w, h, output, frames = predict_vid('vid.mp4', model)

    # the output of the model is
    # a lot of images and for each
    # one a lot of boxes so we have
    # to filter them and get the
    # best predictions
    idxes = []
    images = []
    for img in output:
        boxes, idxs = get_boxes(img, w, h)
        idxes.append(idxs)
        images.append(boxes)

    finished_frames = []
    # draw the boxes on the frames
    for idxs, boxes, frame in zip(idxes, images, frames):
        finished_frame = show_image(frame, boxes, idxs, show=False)
        finished_frames.append(finished_frame)

    count = 0
    for frame in finished_frames:
        cv2.imwrite("frames/%07d.jpg" % count, frame)  # save frame as JPEG file
        count += 1



    #creating video from the frames that have been saved
    fps, duration = 24, 5
    subprocess.call(
        ["ffmpeg", "-y", "-r"
            , str(fps), "-i"
            , "frames/%07d.jpg"
            , "-vcodec", "mpeg4"
            , "-qscale", "5", "-r"
            , str(fps), "stream/video.mp4"])

    # cleaning the frames folder
    clean_frames_folder()
