import pickle
import cv2
import numpy as np

def check():
    cap=cv2.VideoCapture(1)
    if cap.read()[0]==False:
        cap = cv2.VideoCapture(0)
    flag=False
    while True:
        img=cv2.flip(cap.read()[1],1)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)       
        input=cv2.waitKey(1)
        x, y, w, h = 420, 140, 10, 10
        d=10
        img_hand = None
        for i in range(10):
            for j in range(5):
                if np.any(img_hand == None):
                    img_hand = img[y:y+h, x:x+w]
                else:
                    img_hand = np.vstack((img_hand, img[y:y+h, x:x+w]))
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                x+=w+d
            x = 420
            y+=h+d          
        if input == 49:     
            flag=True
            hsv_hand=cv2.cvtColor(img_hand,cv2.COLOR_BGR2HSV)
            hand_histogram=cv2.calcHist([hsv_hand],[0,1],None,[180,256], [0, 180, 0, 256])
            cv2.normalize(hand_histogram,hand_histogram,0,255,cv2.NORM_MINMAX)
        elif input == 50:
            break   
        if flag:
            dst = cv2.calcBackProject([hsv], [0, 1], hand_histogram, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst,-1,disc,dst)
            blur = cv2.GaussianBlur(dst, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh,thresh,thresh))
            cv2.imshow("Thresh", thresh)    
        cv2.imshow("Set hand histogram", img) 
    cap.release()
    cv2.destroyAllWindows()
    with open("histogram1","wb") as f:
        pickle.dump(hand_histogram,f)
    
check()   
    