import numpy as np
import cv2
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    ################### BORDER!!!!!!###################################
    imgL = cv2.copyMakeBorder(Il, 1 , 1 , 1 , 1, cv2.BORDER_CONSTANT, value=0)
    imgR = cv2.copyMakeBorder(Ir, 1 , 1 , 1 , 1, cv2.BORDER_CONSTANT, value=0)
    ################## ARRAY FOR SAVE HAMMING #########################
    # 9 point per pixel
    imgL_b = np.zeros((9, *imgL.shape))
    imgR_b = np.zeros((9, *imgR.shape))
    idx = 0
    # rolling with 3*3 window
    
    ######
    ###### set #(1,1) as (0,0)
    ######
    
    for x in range(-1, 2):
        for y in range(-1, 2):  # 1 if the value of center if greater than neighbor, else 0.
            maskL = (imgL > np.roll(imgL, [y, x], axis=[0, 1]))
            imgL_b[idx][maskL] = 1
            maskR = (imgR > np.roll(imgR, [y, x], axis=[0, 1]))
            imgR_b[idx][maskR] = 1
            idx += 1

    # remove the padded
    imgL_b = imgL_b[:, 1:-1, 1:-1] 
    imgR_b = imgR_b[:, 1:-1, 1:-1]
    #print(L_Bpl.shape)
    #print(R_Bpl.shape)
    #L_Bpl = np.flip(np.reshape(L_Bpl,*imgL.shape).T)
    #R_Bpl = np.flip(np.reshape(R_Bpl,*imgR.shape).T)
    
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    # create cost volumes of shape (h,w,N+1) to record Hamming distances under N+1 disparity
    l_cost_v = np.zeros((max_disp+1, h, w))
    r_cost_v = np.zeros((max_disp+1, h, w))

    wndw_size = -1 # if it is non-positive, it is computed from sigmaSpace
    sigma_r, sigma_s= 4, 11 # parameter for Joint bilateral filter
    for d in range(max_disp+1):
        l_shift = imgL_b[:, :, d:].astype(np.uint32)  
        r_shift = imgR_b[:, :, :w-d].astype(np.uint32) 
        #print(l_shift.shape)
        #print(r_shift.shape)
        
        # compute Hamming distance by xor(a ,b): "^" 
        cost = np.sum((l_shift ^ r_shift), axis=0)
        
        cost = np.sum(cost, axis=2).astype(np.float32) # sum up costs of different channels
        #print(cost.shape)
        #print(Il.shape)
        # left-to-right check
        l_cost = cv2.copyMakeBorder(cost, 0, 0, d, 0, cv2.BORDER_REPLICATE)  # fill left border with border_replicate
        #print(l_cost.shape)
        l_cost_v[d] = xip.jointBilateralFilter(Il, l_cost, wndw_size, sigma_r, sigma_s)
        # right-to-left check
        r_cost = cv2.copyMakeBorder(cost, 0, 0, 0, d, cv2.BORDER_REPLICATE)  # fill right border with border_replicate
        r_cost_v[d] = xip.jointBilateralFilter(Ir, r_cost, wndw_size, sigma_r, sigma_s)
    #cv2.imwrite("cost.png",r_cost[250:200])
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all

    # enhance the disparity map
    l_disp_map = np.argmin(l_cost_v, axis=0)
    r_disp_map = np.argmin(r_cost_v, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    # consistency check
    lr_check = np.zeros((h, w), dtype=np.float32)
    x, y = np.meshgrid(range(w),range(h))
    r_x = (x - l_disp_map) # x-DL(x,y)
    mask1 = (r_x >= 0) # coordinate should be non-negative integer
    l_disp = l_disp_map[mask1]
    r_disp = r_disp_map[y[mask1], r_x[mask1]]
    mask2 = (l_disp == r_disp) # check if DL(x,y) = DR(x-DL(x,y))
    lr_check[y[mask1][mask2], x[mask1][mask2]] = l_disp_map[mask1][mask2]
    
    # hole filling
    l_labels = np.zeros((h, w), dtype=np.float32)
    r_labels = np.zeros((h, w), dtype=np.float32)
    # pad maximum disparity for the holes in boundary
    lr_check_pad = cv2.copyMakeBorder(lr_check,0,0,1,1, cv2.BORDER_CONSTANT, value=max_disp)
    for y in range(h):
        for x in range(w):
            idx_L, idx_R = 0, 0
          
            while lr_check_pad[y, x+1-idx_L] == 0:  # left search 
                idx_L += 1
            l_labels[y, x] = lr_check_pad[y, x+1-idx_L]
            while lr_check_pad[y, x+1+idx_R] == 0:  # right search 
                idx_R += 1
            r_labels[y, x] = lr_check_pad[y, x+1+idx_R]

    # filled disparity map 𝐷 = min(𝐹𝐿 , 𝐹𝑅) (pixel-wise minimum)
    labels = np.min((l_labels, r_labels), axis=0)
    
    # weighted median filter
    WMF_r = 11
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, WMF_r)

    return labels.astype(np.uint8)