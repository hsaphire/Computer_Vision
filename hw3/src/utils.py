import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0] #input xy destination 
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2*N,9))
    for i in range(N):
        A[i*2,:] = [u[i][0],u[i][1],1,0,0,0,-1*u[i][0]*v[i][0],-1*u[i][1]*v[i][0],-v[i][0]]
        A[i*2+1,:] = [0,0,0,u[i][0],u[i][1],1,-1*u[i][0]*v[i][1],-1*u[i][1]*v[i][1],-v[i][1]]
    # TODO: 2.solve H with A
        [_,_,vt]=  np.linalg.svd(A)
        
        h = vt.T[:,-1]
        H= np.reshape(h,(3,3))
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xx ,yy = np.meshgrid(range(xmin,xmax),range(ymin,ymax))
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    U = np.concatenate(([xx.reshape(-1)],[yy.reshape(-1)],
                        [np.ones((xmax-xmin)*(ymax-ymin))]),axis=0)  #for normalize
    k = np.concatenate(([[xx.reshape(-1)]],[[yy.reshape(-1)]]),axis=0)
    
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        H_inv = np.linalg.inv(H)
        V = H_inv@ U #dst axis
        V = V/V[2]  #normalization
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        
        Vx = V[0].reshape(ymax-ymin,xmax-xmin) #reshape 2 source.shape
        Vy = V[1].reshape(ymax-ymin,xmax-xmin)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        mask = (((Vx<w_src-1) & (0<=Vx)) & ((Vy<h_src-1) & (0<=Vy)))
        '''
        print(Vx.shape)
        print(ymax-ymin)
        print(mask.shape)
        print(dst[ymin:ymax,xmin:xmax].shape)#(x)
        print(dst[0:h_dst,0:w_dst].shape)
        '''
        Vx_mask = Vx[mask]
        Vy_mask = Vy[mask]
        
        bVx = (Vx_mask).astype(int)
        bVy = (Vy_mask).astype(int)
        # TODO: 6. assign to destination image with proper masking
        plot = bilinear_interpolation(src,Vx_mask,Vy_mask,bVx,bVy)
       
        #print(mask.shape)#(V)
        
        
        dst[ymin:ymax,xmin:xmax][mask] = plot[bVy,bVx]
        
        

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = H@U
        
        V = (V/V[2]).astype(int) #normalize
        
        Vx = V[0].reshape(ymax-ymin,xmax-xmin)
        Vy = V[1].reshape(ymax-ymin,xmax-xmin)
       
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = ((Vx<w_dst)&(0<=Vx))&((Vy<h_dst)&(0<=Vy))
        # TODO: 5.filter the valid coordinates using previous obtained mask
        Vx_mask = Vx[mask]
        Vy_mask = Vy[mask]
        # TODO: 6. assign to destination image using advanced array indicing
        dst[Vy_mask,Vx_mask,:] = src[mask]
        

    return dst

def bilinear_interpolation(src,Vx,Vy,bVx,bVy):
    plot = np.zeros((src.shape))
    #print("plot",plot.shape)
    a = (Vx-bVx).reshape((-1,1))
    b = (Vy-bVy).reshape((-1,1))
    #print(bVy+1)
    
    plot[bVy,bVx,:] += (1-a)*(1-b)*src[bVy+1,bVx,:]
    plot[bVy,bVx,:] += b*(1-a)*src[bVy+1,bVx,:]
    plot[bVy,bVx,:] += a*b*src[bVy+1,bVx+1,:]
    plot[bVy,bVx,:] += (1-(b))*a*src[bVy,bVx+1,:]

    return plot

