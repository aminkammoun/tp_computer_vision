import cv2 
import numpy as np
path = r'/home/professor_/education/computer vision/bird.jpg'
image = cv2.imread(path)
window_name = 'image'
window_name_flipped = 'image fliped'
window_name_resize = 'image resized'
cv2.imshow(window_name,image)
cv2.waitKey()
height, width = image.shape[:2]
center = (width/2, height/2)
def image_augmentation(image, path):
    imageflip = cv2.flip(image, 0)
    cv2.imshow(window_name,imageflip)
    resizedimage= cv2.resize(image,(200,200),interpolation = cv2.INTER_AREA)
    M = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)
    rotated_image = cv2.warpAffine(src=image, M=M, dsize=(width, height))
    matrix = np.ones(image.shape, dtype = "uint8") * 120
    imagebrightness = cv2.subtract(image, matrix)
    Kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    cv2.imshow(window_name,sharpened)
    #cv2.imshow(window_name,rotated_image)
    cv2.waitKey()

image_augmentation(image,path)