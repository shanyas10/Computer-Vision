import numpy as np
import cv2


def order_points(pts):
    rect = np.zeros((4,2), dtype ="float32") # TL:1, TR:2, BR:3, BL:4
    
    s= pts.sum(axis=1)
    print(s)
    
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    print(diff)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA   = np.sqrt(((br[0] - tr[0])**2) + ((br[1] - tr[1])**2))
    heightB   = np.sqrt(((bl[0] - tl[0])**2) + ((bl[1] - tl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0,0],
        [maxWidth - 1 , 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    #compute perspective transform and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped 


    