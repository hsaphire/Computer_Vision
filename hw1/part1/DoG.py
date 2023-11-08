import numpy as np
import cv2
import os
class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image,save =False):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        gaussian_images_r = []
        image_shape = image.shape
        gaussian_images.append(image.copy())        
        
        def Gaussian(list):
            for k in range(1,self.num_guassian_images_per_octave): 
                list.append(cv2.GaussianBlur (list[0],(0, 0), self.sigma**k).copy())
        
        def Merge(list):
            merge_res =[]
            for k in range(0,len(list)-2):
                merge_res.append(cv2.merge([list[k],list[k+1],list[k+2]]).copy())
            
            return merge_res
        
        def extremum_filter(image):
            shape = image.shape
            result = np.zeros([shape[0],shape[1],3])
            for i in range(1,shape[0]):
                for j in range(1,shape[1]):
                 
                    if image[i,j,1] <= image[i-1:i+2,j-1:j+2,:].min() :
                        result[i,j,1] =  image[i,j,1]
                    elif image[i,j,1] >= image[i-1:i+2,j-1:j+2,:].max() :
                            result[i,j,1] =  image[i,j,1]       

            return result
        
        #計算4維陣列極值 輸出有兩個filter[0,1]
        def Find_4axis(mer):
            filter = []
            for item,scale in enumerate (mer):
                
                filter.append(extremum_filter(scale).copy())
            return filter
        
        def threshold (image):
            img = image[:,:,1]
            array =  []
            #print(img.max(),img.min())
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if abs(img[i][j]) > self.threshold:

                        array.append((i,j))
            return np.array(array)
        def saveimg(img,key):
            if save:
                
                if not os.path.exists("out"):
                    os.mkdir("out")
                
                for i in range(len(img)):
                    out = img[i].copy()
                    out = ((out - out.min()) / (out.max()-out.min()) *255).round()
                    cv2.imwrite("out/"+key+str(i+1) +".png", out)

        
        Gaussian(gaussian_images)
        gaussian_images_r.append(cv2.resize(gaussian_images[4], (image_shape[1]//2, image_shape[0]//2), interpolation=cv2.INTER_NEAREST).copy())
        Gaussian(gaussian_images_r)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        dog_images_r = []
        

        for j in range(self.num_DoG_images_per_octave):
            dog_images.append((cv2.subtract(gaussian_images[j+1], gaussian_images[j])).copy())
            dog_images_r.append((cv2.subtract(gaussian_images_r[j+1], gaussian_images_r[j])).copy())
        
        merge1 = Merge(dog_images)
        merge2 = Merge(dog_images_r)
       
        saveimg(dog_images, "DoG1-")
        saveimg(dog_images_r, "DoG2-")
        

        F1 = Find_4axis(merge1)
        F2 = Find_4axis(merge2)
        '''
        for c,d in enumerate(F1):
            cv2.imshow('My Image', d.astype(np.uint8))
            print(d[:,:,1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''
       
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        
        t1 = [threshold(F1[0]),threshold(F1[1]),threshold(F2[0])*2,threshold(F2[1])*2]
        
        local_extremum = np.zeros([1,2]).astype("int32")
        true = 0
        for item,zero in enumerate (t1):
            if zero != []:
                if true == 0:
                    ex = zero
                    true =1
                else :
                    local_extremum = np.concatenate((ex,zero),axis=0)
                    #print(local_extremum.shape)
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        #print(local_extremum)
        keypoints = np.unique(local_extremum,axis=0)
        #print(keypoints)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
    
        return keypoints
