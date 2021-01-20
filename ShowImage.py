try:
    import cv2
except ImportError:
    print('\nOpenCV-python is uninstalled')


def show_image(image_path, boxes, idxs, show):
    '''

    :param idxs: indexes of the proper boxes

    :param image_path: if we in the part of the image then
    this parameter is the path to the image we want to predict
    its boxes
    if we in the part of the video then this parameter is a matrix
    that represent a frame

    :param boxes:the boxes that we want to draw on the images

    :param show: if we in the part of the image then the parameter is true
    if we in the part of the video then this parameter is false in order not to show the frame

    the function return the frame with the boxes on the frame

    '''
    if len(idxs) > 0:
        if show:
            image = cv2.imread(image_path)
        else:
            image=image_path
        # loop over the indexes we are keeping
        for i in boxes:
            # extract the bounding box coordinates
            (x, y) = (i[0], i[1])
            (w, h) = (i[2], i[3])
            # draw a bounding box rectangle and label on the image
            color = [0, 0, 255]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(i[4])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if show:
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    return image
