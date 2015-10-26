#
# Hand detection
#
# Requires OpenCV2 2.0 and Python 2.6
#

import cv2
import time


def get_img(capture):
    img = cv2.QueryFrame(capture)
    return img

def get_hands(image):
    """ Returns the hand as white on black. Uses value in HSV to determine
        hands."""
    size = cv2.GetSize(image)
    hsv = cv2.CreateImage(size, 8, 3)
    hue = cv2.CreateImage(size, 8, 1)
    sat = cv2.CreateImage(size, 8, 1)
    val = cv2.CreateImage(size, 8, 1)
    hands = cv2.CreateImage(size, 8, 1)
    cv2.Cv2tColor(image, hsv, cv2.CV2_BGR2HSV)
    cv2.Split(hsv, hue, sat, val, None)

    cv2.ShowImage('Live', image)
    cv2.ShowImage('Hue', hue)
    cv2.ShowImage('Saturation', sat)

    cv2.Threshold(hue, hue, 10, 255, cv2.CV2_THRESH_TOZERO) #set to 0 if <= 10, otherwise leave as is
    cv2.Threshold(hue, hue, 244, 255, cv2.CV2_THRESH_TOZERO_INV) #set to 0 if > 244, otherwise leave as is
    cv2.Threshold(hue, hue, 0, 255, cv2.CV2_THRESH_BINARY_INV) #set to 255 if = 0, otherwise 0
    cv2.Threshold(sat, sat, 64, 255, cv2.CV2_THRESH_TOZERO) #set to 0 if <= 64, otherwise leave as is
    cv2.EqualizeHist(sat, sat)

    cv2.Threshold(sat, sat, 64, 255, cv2.CV2_THRESH_BINARY) #set to 0 if <= 64, otherwise 255

    cv2.ShowImage('Saturation threshold', sat)
    cv2.ShowImage('Hue threshold', hue)

    cv2.Mul(hue, sat, hands)

    #smooth + threshold to filter noise
#    cv2.Smooth(hands, hands, smoothtype=cv2.CV2_GAUSSIAN, param1=13, param2=13)
#    cv2.Threshold(hands, hands, 200, 255, cv2.CV2_THRESH_BINARY)

    cv2.ShowImage('Hands', hands)

    return hands

if __name__ == "__main__":
    cv2.NamedWindow('Live', cv2.CV2_WINDOW_AUTOSIZE)

    #set up connection to camera
    capture = cv2.CaptureFromCAM(0)
    if not capture:
        print "Error opening capture device"
        sys.exit(1)

    cv2.SetCaptureProperty(capture, cv2.CV2_CAP_PROP_FRAME_WIDTH, 320)
    cv2.SetCaptureProperty(capture, cv2.CV2_CAP_PROP_FRAME_HEIGHT, 240)

    while 1:
        image = get_img(capture)
        hands = get_hands(image)

        # handle events
        k = cv2.WaitKey(5)