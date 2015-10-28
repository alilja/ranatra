import numpy as np
import cv2

cam = cv2.VideoCapture(0)


class HandDetection:
    def __init__(self):
        self.trained_hand = False
        self.hand_row_nw = None
        self.hand_row_se = None
        self.hand_col_nw = None
        self.hand_col_se = None
        self.hand_hist = None

    def draw_hand_rect(self, frame):
        rows, cols, _ = frame.shape

        self.hand_row_nw = np.array([
            6 * rows / 20,
            6 * rows / 20,
            6 * rows / 20,
            10 * rows / 20,
            10 * rows / 20,
            10 * rows / 20,
            14 * rows / 20,
            14 * rows / 20,
            14 * rows / 20
        ])
        self.hand_col_nw = np.array([
            3 * cols / 20,
            5 * cols / 20,
            7 * cols / 20,
            3 * cols / 20,
            5 * cols / 20,
            7 * cols / 20,
            3 * cols / 20,
            5 * cols / 20,
            7 * cols / 20
        ])

        self.hand_row_se = self.hand_row_nw + 10
        self.hand_col_se = self.hand_col_nw + 10

        size = self.hand_row_nw.size
        for i in xrange(size):
            cv2.rectangle(frame, (
                self.hand_col_nw[i],
                self.hand_row_nw[i]),
                (self.hand_col_se[i], self.hand_row_se[i]),
                (0, 255, 0),
                1
            )
        black = np.zeros(frame.shape, dtype=frame.dtype)
        frame_final = np.vstack([black, frame])
        return frame

    def train_hand(self, frame):
        self.set_hand_hist(frame)
        self.trained_hand = True

    def set_hand_hist(self, frame):
        # TODO use constants, only do HSV for ROI
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([90, 10, 3], dtype=hsv.dtype)

        size = self.hand_row_nw.size
        for i in xrange(size):
            roi[i * 10:i * 10 + 10, 0:10] = hsv[self.hand_row_nw[i]:self.hand_row_nw[i] + 10, self.hand_col_nw[i]:self.hand_col_nw[i] + 10]

        self.hand_hist = cv2.calcHist([roi], [0, 1], None, [30, 256], [0, 30, 0, 256])
        cv2.normalize(self.hand_hist, self.hand_hist, 0, 255, cv2.NORM_MINMAX)


def apply_hist_mask(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 30, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 20, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))



    res = cv2.bitwise_and(frame, thresh)
    return res


def calc_max_contour(contours):
    max_i = 0
    max_area = 0

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_i = i

    contour = contours[max_i]
    return contour


def calc_hull(contour):
    hull = cv2.convexHull(contour)
    return hull


def calc_defects(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is not None and len(hull > 3) and len(contour) > 3:
        defects = cv2.convexityDefects(contour, hull)
        return defects
    else:
        return None


def calc_centroid(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    else:
        return None


def calc_farthest_point(defects, contour, centroid):
    s = defects[:, 0][:, 0]
    cx, cy = centroid

    x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
    y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

    xp = cv2.pow(cv2.subtract(x, cx), 2)
    yp = cv2.pow(cv2.subtract(y, cy), 2)
    dist = cv2.sqrt(cv2.add(xp, yp))

    dist_max_i = np.argmax(dist)

    if dist_max_i < len(s):
        farthest_defect = s[dist_max_i]
        farthest_point = tuple(contour[farthest_defect][0])
        return farthest_point
    else:
        return None

hand_detection = HandDetection()

while(1):
    rep, frame = cam.read()
    rows, cols, _ = frame.shape
    frame = frame[:, :cols / 2]

    if not hand_detection.trained_hand:
        frame = hand_detection.draw_hand_rect(frame)

        if cv2.waitKey(1) == ord('h') & 0xFF:
                hand_detection.train_hand(frame)
    else:
        tracking_frame = apply_hist_mask(frame, hand_detection.hand_hist)
        cv2.GaussianBlur(tracking_frame, (13, 13), 0, tracking_frame)
        gray = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours is not None and len(contours) > 0:
            cv2.drawContours(frame, contours, -1, (0,255,0), 3)

            max_contour = calc_max_contour(contours)
            hull = calc_hull(max_contour)
            centroid = calc_centroid(max_contour)
            defects = calc_defects(max_contour)

            if centroid is not None and defects is not None and len(defects) > 0:
                farthest_point = calc_farthest_point(defects, max_contour, centroid)

                if farthest_point is not None:
                    cv2.circle(frame, farthest_point, 5, [0, 0, 255], -1)

    cv2.imshow('image', frame)

cam.release()
cv2.destroyAllWindows()