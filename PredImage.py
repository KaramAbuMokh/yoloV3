import cv2


def read_image(image_path):
    '''
    reding image from file.
    :param image_path: the path to the image.
    :return: width, height and the image values as matrix.
    '''
    image = cv2.imread(image_path)
    (w, h) = image.shape[:2]
    return w,h,image


def predict_image(image_path,model):
    '''
    predict bounding boxes using the model.
    :param image_path: path to the image we want to detect object in it.
    :param model: the model that will detect the objects.
    :return: the output of the model .
    will process the output after that.
    '''
    w,h,image=read_image(image_path)

    # determine only the *output* layer names that we need from YOLO
    ln = model.getLayerNames()
    ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    model.setInput(blob)
    output = model.forward(ln)
    return w,h,output
