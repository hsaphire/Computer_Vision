import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=1.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='testdata/2.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path,0).astype(np.float32)
    #print(img)
    ### TODO ###
    DOg = Difference_of_Gaussian(args.threshold)
    keypoint = DOg.get_keypoints(img,save=True)
    '''
    for id,imgo in enumerate (dog1) :
        
        imgo = ((imgo - imgo.min()) / (imgo.max()-imgo.min())*255).round()
        cv2.imwrite("out/"+args.image_path[9]+"/"+"DoG"+str(id)+".png",imgo)

        #assert False
        imgr = ((dogr[id] - dogr[id].min()) / (dogr[id].max()-dogr[id].min())*255).round()
        cv2.imwrite("out/"+args.image_path[9]+"/"+"DoG_r"+str(id)+".png",imgr)
    '''

    
    plot_keypoints(img, keypoint, "out/test2/"+"keypoint"+str(args.threshold)+".png")
if __name__ == '__main__':
    main()