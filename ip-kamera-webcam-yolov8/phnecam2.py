import cv2

#Starting the video capture

cap = cv2.VideoCapture("http://192.168.1.22:8080//video")
 
while(cap.isOpened()):
    ret,img = cap.read()

    #Controlling the algorithm with keys
    try:
        img = cv2.resize(img,(640,480))
        cv2.imshow('img',img)
        a = cv2.waitKey(1)
        if a == ord('q'):
            break
    except cv2.error:
        print("stream ended")
        break

cap.release()
cv2.destroyAllWindows()
