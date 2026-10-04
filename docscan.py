import cv2 as cv
img=cv.imread('C:/Users/HP/Pictures/cat.jpeg')

#cv.imshow("cat",img)

def rescaleframe(frame , scale=0.75):
    width= int(frame.shape[1]*scale)
    height= int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions , interpolation=cv.INTER_AREA)


resizedimage=rescaleframe(img,0.2)
cv.imshow("image",resizedimage)
cv.waitKey(0)







