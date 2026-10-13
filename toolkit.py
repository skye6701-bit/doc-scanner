import cv2 as cv
import os
from dotenv import load_dotenv

load_dotenv()

image_address=os.getenv("image_address")

img=cv.imread(image_address)

if img is None:
    raise FileNotFoundError(f"Could not read image at: {image_address}")

cv.imshow("cat",img)

#CREATING THE FUNCTION FOR RESCALE
def rescaleframe(frame , scale=0.50):
    width= int(frame.shape[1]*scale)
    height= int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions , interpolation=cv.INTER_AREA)

#USING THE FUNCTION FOR RESIZE
resizedimage=rescaleframe(img,0.2)

cv.imshow("image",resizedimage)

#conerting the image into gray scale
gray=cv.cvtColor(resizedimage,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

#converting the image into blur
blur=cv.GaussianBlur(resizedimage,(3,3),cv.BORDER_DEFAULT)
cv.imshow("blur",blur)

#FINDING IMAGES
canny=cv.Canny(resizedimage,125,175)
cv.imshow('image',canny)

#translate function


cv.waitKey(0)
