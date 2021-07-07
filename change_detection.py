import glob
import cv2
import numpy as np

path = glob.glob("C:/Users/pc/Desktop/input/*.jpg")
path2 = glob.glob("C:/Users/pc/Desktop/output/*.jpg")


'''
def dilation_erosion(image):
    kernel =np.ones((3,3), np.uint8)
    image=cv2.dilate(image,kernel, iterations=1)
    image=cv2.erode(image,kernel, iterations=1)
    return image
'''

def modified_median(source):
    final = source[:]
    for y in range(len(source)):
        for x in range(y):
            final[y,x]=source[y,x]

    members=[source[0,0]]*9
    for y in range(1,source.shape[0]-1):
        for x in range(1,source.shape[1]-1):
            members[0] = source[y-1,x-1]
            members[1] = source[y,x-1]
            members[2] = source[y+1,x-1]
            members[3] = source[y-1,x]
            members[4] = source[y,x]
            members[5] = source[y+1,x]
            members[6] = source[y-1,x+1]
            members[7] = source[y,x+1]
            members[8] = source[y+1,x+1]
            members.sort()
            final[y,x]=members[5]
    return final


if __name__=='__main__':

    #input images load(gray scale)
    images = []
    for img in path:
        n = cv2.imread(img)
        n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        images.append(n)
    
    #BG Subtraction Setting
    #bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    #history = 5    
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
    
    background=cv2.medianBlur(images[177],3)
    
    
    #Decide how long to repeat from where
    #start_frame_num=800
    #repeat_num=200
    start_frame_num=1
    repeat_num=3890
     
    image = images[start_frame_num]
   
    
    image_frame=start_frame_num
    count=0
    
    #morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    
    #kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    
    while True:
        #BG Subtraction
        image=cv2.absdiff(image,background)
        
        #BG
        #image=bgSubtractor.apply(image,learningRate=1.0/history)
        #image=bgSubtractor.apply(image)
        
        #median filter
        #image = modified_median(image)
        image = cv2.medianBlur(image, 7)     
        
        #binary map
        ret, image = cv2.threshold(image, 36, 255, cv2.THRESH_BINARY)
        #image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,10)
        #image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,21,5)
        
        #morphological
        image=cv2.morphologyEx(image, cv2.MORPH_CLOSE,kernel)
        image=cv2.morphologyEx(image, cv2.MORPH_OPEN,kernel2)
                
       
        #output to monitor 
        cv2.imshow("Object Movement", image)
        
        #output to image file
        cv2.imwrite('C:/Users/pc/Desktop/output/{}.jpg'.format(start_frame_num+count), image) 

        image_frame=image_frame +1
        image=images[image_frame]
        

        #End the processing(ESC)
        key = cv2.waitKey(10)
        if key == 27:
            break
        
        #End the processing(done all frame)
        count=count+1
        if count==repeat_num:
            break
    
        
    cv2.destroyAllWindows()

    
