import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('photo.jpg')

#convert the RGB images to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#scale factor
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

#create a faces array
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

#resized the images
resize = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

#shows the image through the window
cv2.imshow('Images', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
