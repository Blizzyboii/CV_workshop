import cv2

clf = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


camera = cv2.VideoCapture(0)

while True:
  _,frame = camera.read()
  #print(frame.shape)
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces = clf.detectMultiScale(
    frame,
    scaleFactor=1.1,
    minNeighbors=5,  # higher the number, more strict it becomes
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
  )

  for (x,y,width,height) in faces:
     print(faces)
     cv2.rectangle(frame,(x,y),(x+width,y+height),(255,255,0), 2)
     roi_gray = gray[y:y+height, x:x+width]
     roi_color = frame[y:y+height, x:x+width]
     eyes = eye_cascade.detectMultiScale(
        roi_color,
        minNeighbors=5)
     for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

  cv2.imshow('Faces',frame)

  if cv2.waitKey(1) == ord("q"):
     break
  
camera.release()
cv2.destroyAllWindows()
