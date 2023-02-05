import cv2         #opencv library ,open sourse 
face_cap = cv2.CascadeClassifier( "C:/python11/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml" )# capture the face feature 

vid_cap = cv2.VideoCapture(0)#for capturing the data
while True:
    ret , video_data = vid_cap.read() # to read data
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY) 
    faces = face_cap.detectMultiScale(  #  it will capture the face data
        col,                                  
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:            #for rectangle border 
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live", video_data)#show the data & name as video_live
    if cv2.waitKey(10) == ord ("c"): #to close it 
        break
vid_cap.release()# to release the resourses
