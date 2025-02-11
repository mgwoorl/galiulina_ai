import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

haar_cascade = "haarcascades/haarcascade_eye_tree_eyeglasses.xml"

eyes = cv2.CascadeClassifier(haar_cascade)

sunglasses = cv2.imread("dealwithit.png")
sunglasses_grey = cv2.cvtColor(sunglasses, cv2.COLOR_BGR2GRAY)

def detector(img, classifier, scaleFactor=None, minNeighbors=None):
    result = img.copy()
    rects = classifier.detectMultiScale(result, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # for (x, y, w, h) in rects:
    #     cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255))

    if len(rects) < 2:
        return result

    if rects[0][0] < rects[1][0]:
        left_eye = rects[0]
        right_eye = rects[1]

    else:
        left_eye = rects[1]
        right_eye = rects[0]

    x1_sunglasses = left_eye[0]
    x2_sunglasses = right_eye[0] + right_eye[2]

    y1_sunglasses = left_eye[1] - left_eye[3] // 4
    y2_sunglasses = left_eye[1] + left_eye[3]

    width = x2_sunglasses - x1_sunglasses
    height = int(1.2 * (y2_sunglasses - y1_sunglasses))
    y1_sunglasses = y1_sunglasses - int(0.1 * (y2_sunglasses - y1_sunglasses))

    sunglasses_resized = cv2.resize(sunglasses, (width, height))

    if x1_sunglasses < 0 or y1_sunglasses < 0 or x1_sunglasses + width > img.shape[1] or y1_sunglasses + height > img.shape[0]:
        return result

    roi = result[y1_sunglasses:y1_sunglasses + height, x1_sunglasses:x1_sunglasses + width]
    sunglasses_gray = cv2.cvtColor(sunglasses_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(sunglasses_gray, 1, 255, cv2.THRESH_BINARY_INV)
    fg = cv2.bitwise_and(sunglasses_resized, sunglasses_resized, mask=mask)
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    combined = cv2.add(bg, fg)

    result[y1_sunglasses:y1_sunglasses + height, x1_sunglasses:x1_sunglasses + width] = combined

    return result

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow("Camera", detector(frame, eyes, 1.2, 5))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
