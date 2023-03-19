import cv2
import imutils
import pytesseract #to convert from img to text

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

# Read image, resize it then print it

image = cv2.imread('Car.jpg')
image = imutils.resize(image, width=1000)
cv2.imshow("original", image)
cv2.waitKey(0)     # waits for user to press a key before continuing execution

# Converting to a Greyscale image

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("grayed", gray_image)

cv2.waitKey(0)

# Smoothening the image

gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("smoothened", gray_image)
cv2.waitKey(0)

# Showing the edges of the feature in the image

edged = cv2.Canny(gray_image, 30, 200)
cv2.imshow("edged", edged)
cv2.waitKey(0)

# Highlighting the contour lines of the feature in the image(Curve containing points of similar colour/intensity)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("contours", image1)
cv2.waitKey(0)

# Highlighting the significant contours in the image

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 contours", image2)
cv2.waitKey(0)
i = 7
for c in cnts:                                              # for loop across contours found
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)  # approximating shape of contour
    if len(approx) == 4:
        screenCnt = approx
    x, y, w, h = cv2.boundingRect(c)               # finds coordinates of identified shape
    new_img = image[y:y + h, x:x + w]
    cv2.imwrite('./' + str(i) + '.png', new_img)    # storing new image of cropped image
    i += 1
    break
gray_image2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_image2, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow("Image with detected license plate", image)
cv2.waitKey(0)
Cropped_loc = './7.png'
cv2.imshow("cropped", cv2.imread(Cropped_loc))
plate = pytesseract.image_to_string(Cropped_loc, lang='eng',config ='--psm 13')
print("Number plate is:", plate)
cv2.waitKey(0)
cv2.destroyAllWindows()

