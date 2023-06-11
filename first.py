import cv2
import imutils #resize image
import pytesseract

pytesseract.pytesseract.tesseract_cmd=r"D:\\Tesseract\\tesseract.exe"

# read image files
image = cv2.imread('D:\License plate detection 2\img_dataset\car4.jpeg')

# resizing and standardize images to  300
image = imutils.resize(image, width=300)

# We will display orignal image when it will start finding
cv2.imshow("Original Image",image)
cv2.waitKey(0) #till i press anything it will not execute further

#converting image to grayscale as it reduces dimensions & complexity and also some algo like canny only works on gray img
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image",gray)
cv2.waitKey(0)

# now we will reduce  noise form our image and make it smooth
gray = cv2.bilateralFilter(gray, 11 , 17, 17)
cv2.imshow("Smoother Image", gray)
cv2.waitKey(0)

# now we will  find the edges of images
edged = cv2.Canny(gray , 170, 200)
cv2.imshow("Canny edge", edged)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)


# ok so here 'cnts' is   contours(green lines) Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
# new -,is heirarchy - relationship
# RETR_LIST - it takes all the contours but dosn't create any parent-child relationship
# CHAIN_APPROX_SIMPLE - it removes all the redundant points and compress the contour by saving memory as we dont need all the points to recognixe shape so only specific points are taken by it
                        # ex - it takes only 4 points of the square(Number plate) rather than whole 4 side line 

# now , we will create a copy of our original image to draw all the contours
# -1 signifies drawing all contours
image1 = image.copy()
cv2.drawContours(image1 , cnts, -1 , (0,255,0), 3 ) #This values r fixed to draw all contours in immage
cv2.imshow('Canny after conturing', image1)
cv2.waitKey(0)


#  We dont want all contours , only interested in number plates
#  but can’t directly locate that ,so we will sort them on the basis of their areas
#  we will select those ares which are maximun, so we will select top 30 areas
#  but it will give sorted list as in order of min to maximun
#  so for that we will reverse the order of sorting

cnts = sorted(cnts , key = cv2.contourArea, reverse = True)[:30]
NumberPlatecount = None

#  because currently we don't have any contour or you can say it will show how many number plates are their in image
#  to draw top 30 contours we will make copy of original image and use as  we don't went to edit anything in our original image

image2 = image.copy()
cv2.drawContours(image2 , cnts, -1 , (0,255,0), 3 )
cv2.imshow("TOP 30 Contours", image2)
cv2.waitKey(0)


#  now we will iterate over all the contours and select best possible the contours of our num plate
count = 0
name = 1  #name of our cropped image

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    # perimeter is also called arclength / len of contur and we can find directly in python using arclength function

    approx = cv2.approxPolyDP(i, 0.018 * perimeter, True)
    #  approxPolyDP we have used as it approximates the curve of polygon with precision
    #  here we have used 0.018 * perimeter , which is approximation of curve length
    if(len(approx)== 4):  #as num plate has 4 corners
        NumberPlatecount = approx
        # now we will crop rectangle part
        x , y, w, h=cv2.boundingRect(i)
        # x , y , w , h are x , y , width , height of rectangle
        crp_img = image[y:y*h , x:x*w]
        cv2.imwrite(str(name)+ '.png',crp_img)
        # saving cropped images
        name += 1
        break

# we will draw cnts in main img that we have identified as a num plate
cv2.drawContours(image,[NumberPlatecount], -1 , (0,255,0), 3 )
cv2.imshow("Final image",image)
cv2.waitKey(0)
# we got our num plate img by this 


# now we will crop only the part of num plt

crop_img_loc = '1.png'
cv2.imshow("Cropped Image", cv2.imread(crop_img_loc))
cv2.waitKey(0)

# Now we got our num plt imge and now we will convert our img into text by pytessereact module
text = pytesseract.image_to_string(crop_img_loc, lang ='eng')
print("NUMBER IS : ",text)
cv2.waitKey(0)

