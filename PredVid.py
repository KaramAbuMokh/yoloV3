try:
    import cv2
except ImportError:
    print('\nOpenCV-python is uninstalled')

try:
    import imutils
except ImportError:
    print('\nimutils is uninstalled')

def predict_vid(vid_path,model):
    '''
    predict bounding boxes using the model.
    :param image_path: path to the image we want to detect object in it.
    :param model: the model that will detect the objects.
    :return: the output of the model .
    will process the output after that.
    '''

    ln = model.getLayerNames()
    ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]


    vs = cv2.VideoCapture(vid_path)
    writer = None
    (W, H) = (None, None)

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    frames=[]
    layerOutputs=[]
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        frames.append(frame)
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        model.setInput(blob)
        layerOutputs.append(model.forward(ln))
    return W, H, layerOutputs,frames


