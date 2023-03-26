import re
import cv2
from matplotlib import pyplot as plt
import numpy as np


pts = []  # ROI points

def mouseHandler(event, x, y, flags, param):
    """Mouse handler for selecting ROI, press 'left mouse' to select points, press 'right mouse' to finish

    Args:
        event: event type
        x: x coordinate
        y: y coordinate
    """
    global drawing
    setpoint = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:  # left mouse button to select a point
        drawing = True  # start drawing
        pts.append(setpoint)  # add a point
        print("Select a vertex {}：{}".format(len(pts), setpoint))
    elif event == cv2.EVENT_RBUTTONDOWN:  # right mouse button to finish
        drawing = False  # stop drawing
        print("End of drawing \n ROI Vertex coordinates: {}".format(pts))


def select_roi(img):
    print("Click the left mouse button: Select ROI vertices")
    print("Click the right mouse button: End ROI selection")
    print("Press ESC to exit, if you want global detection, press ESC directly")

    drawing = True  # Turn on drawing status
    cv2.namedWindow('origin')  # create a window
    cv2.setMouseCallback('origin', mouseHandler)  # set mouse callback function
    while True:
        imgCopy = img.copy()
        if len(pts) > 0:
            cv2.circle(imgCopy, pts[-1], 5, (0, 0, 255), -1)  # draw the last point
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.circle(imgCopy, pts[i], 5, (0, 0, 255), -1)  # draw the other points
                    cv2.line(imgCopy, pts[i], pts[i + 1], (255, 0, 0), 2)  # draw the line between two points
            if drawing == False:
                cv2.line(imgCopy, pts[0], pts[-1], (255, 0, 0), 2)  # draw the line between the first point and the last point
        cv2.imshow('origin', imgCopy)
        key = 0xFF & cv2.waitKey(1)  # Press ESC to exit
        if key == 27:
            break
    cv2.destroyAllWindows()  # close all windows
    
    if len(pts) == 0:   # if no points are selected, return the original image
        return np.ones(img.shape[:2], np.uint8) * 255
    else:   # if points are selected, return the ROI image
        points = np.array(pts, np.int32)
        cv2.polylines(img, [points], True, (255, 255, 255), 2)  # draw ROI
        mask = np.zeros(img.shape[:2], np.uint8)  # create a mask
        cv2.fillPoly(mask, [points], (255, 255, 255))
    return mask


if __name__=='__main__':
    # Mouse interaction to draw polygons ROI
    img = cv2.imread("./assets.png")  # read RGB image

    print("Click the left mouse button: Select ROI vertices")
    print("Click the right mouse button: End ROI selection")
    print("Press ESC to exit")

    drawing = True  # Turn on drawing status
    cv2.namedWindow('origin')  # create a window
    cv2.setMouseCallback('origin', mouseHandler)  # set mouse callback function
    while True:
        imgCopy = img.copy()
        if len(pts) > 0:
            cv2.circle(imgCopy, pts[-1], 5, (0, 0, 255), -1)  # draw the last point
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.circle(imgCopy, pts[i], 5, (0, 0, 255), -1)  # draw the other points
                    cv2.line(imgCopy, pts[i], pts[i + 1], (255, 0, 0), 2)  # draw the line between two points
            if drawing == False:
                cv2.line(imgCopy, pts[0], pts[-1], (255, 0, 0), 2)  # draw the line between the first point and the last point
        cv2.imshow('origin', imgCopy)
        key = 0xFF & cv2.waitKey(1)  # Press ESC to exit
        if key == 27:
            break
    cv2.destroyAllWindows()  # close all windows

    points = np.array(pts, np.int32)
    cv2.polylines(img, [points], True, (255, 255, 255), 2)  # draw ROI
    mask = np.zeros(img.shape[:2], np.uint8)  # create a mask
    cv2.fillPoly(mask, [points], (255, 255, 255))
    imgROI = cv2.bitwise_and(img, img, mask=mask)

    plt.figure(figsize=(9, 6))
    plt.subplot(131), plt.title("Origin image"), plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("ROI mask"), plt.axis('off')
    plt.imshow(mask, cmap='gray')
    plt.subplot(133), plt.title("ROI cropped"), plt.axis('off')
    plt.imshow(cv2.cvtColor(imgROI, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
    # plt.savefig("ROI.png")
