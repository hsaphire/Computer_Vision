import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
    cv2.imwrite("out/"+args.image_path[11]+"/gray/img_gray.png",img_gray)
    print("out/"+args.image_path[11]+"/img_gray.png")
    ### TODO ###
    
    #load setting.txt
    setting = open(args.setting_path,'r')
    for time,line in enumerate (setting.readlines()):
        line = line.replace("\n","").split(",")
        rgb = img_rgb.copy()
        if time >=1 and time <=5 :
            img2gray = (rgb[:,:,0]*float(line[0]))+ \
                       rgb[:,:,1]*float(line[1])+ \
                       rgb[:,:,2]*float(line[2]) 

            cv2.imwrite("out/"+args.image_path[11]+"/gray/gray"+str(time)+".png",img2gray)
            
        elif time ==6:
             sigma_r = float(line[3])
             sigma_s = int(line[1])
             print(sigma_r,sigma_s)
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    gray_list = os.listdir('out/'+args.image_path[11]+"/gray")

    for load in gray_list:
        img_gray = cv2.imread(os.path.join('out/'+args.image_path[11]+"/gray/",load),cv2.IMREAD_UNCHANGED)
        #img_gray = img_gray[:,:,0]
        bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        cv2.imwrite("out/"+args.image_path[11]+"/jbf/jbf"+str(load[:-4])+".png",cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR))
        cost = np.sum(abs(np.abs(jbf_out.astype('int32')-bf_out.astype('int32'))))
        print("jbfcost"+load+":",cost)

if __name__ == '__main__':
    main()