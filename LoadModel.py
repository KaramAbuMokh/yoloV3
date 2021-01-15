import cv2

def load_model(configPath,weightsPath):
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net




