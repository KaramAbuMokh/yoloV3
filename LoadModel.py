
try:
    import cv2
except ImportError:
    print('\nOpenCV-python is uninstalled')
def load_model(configPath,weightsPath):
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net




