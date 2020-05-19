import numpy as np
import cv2

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def overlay_image (l_img, s_img, x_offset, y_offset, w, h):
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]
    
    alpha_s = s_img[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 2):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])
    return l_img

sisuga = cv2.imread('data/suga.png')

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = np.copy(colored_img)
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        # cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_copy = overlay_image(img_copy, sisuga, x-10, y-10, w, h)
        
    return img_copy

haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

cam = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cam.read()
    
    img_detected = detect_faces(haar_face_cascade, frame)
    
    cv2.imshow("test", img_detected)
    if not ret:
        break
    k = cv2.waitKey(20)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
