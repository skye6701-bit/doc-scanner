from transform import four_point_transform
import numpy as np
import cv2 as cv
import imutils
import argparse
from skimage.filters import threshold_local

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="C:/Users/HP/Pictures/shopping.jpeg",
                help="Path to input image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)

# FIX 1: Lowered Canny thresholds to detect faint edges against the light table
edged = cv.Canny(gray, 30, 100)

# FIX 2: Added morphological closing to bridge gaps in the detected edges
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
edged = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)

cv.imshow("Image", image)
cv.imshow("Edged", edged)
cv.waitKey(0)
cv.destroyAllWindows()

cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

screenCnt = None
for c in cnts:
    peri = cv.arcLength(c, True)
    # FIX 3: Increased epsilon from 0.02 to 0.05 to forgive tiny bumps along the paper edge
    approx = cv.approxPolyDP(c, 0.05 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    raise RuntimeError("Could not find a document contour (4-sided polygon). Check image lighting or adjust Canny thresholds.")

cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv.imshow("Outline", image)
cv.waitKey(0)
cv.destroyAllWindows()

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

cv.imshow("Original", imutils.resize(orig, height=650))
cv.imshow("Scanned", imutils.resize(warped, height=650))
cv.waitKey(0)
cv.destroyAllWindows()