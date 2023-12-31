import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return H
    if N < 4:
        print('At least 4 points should be given')
        return H

    # TODO: 1.forming A
    A = np.zeros((2*N,9))
    for i in range(N):
        A[i*2,:] = [u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0]*v[i][0], -u[i][1]*v[i][0], -v[i][0]]
        A[i*2+1,:] = [0, 0, 0, u[i][0], u[i][1], 1, -u[i][0]*v[i][1], -u[i][1]*v[i][1], -v[i][1]]
    # TODO: 2.solve H with A
    [_, _, V] = np.linalg.svd(A)
    h = V.T[:,-1]
    H = np.reshape(h,(3,3))
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
    Ux, Uy = np.meshgrid(range(xmin,xmax), range(ymin,ymax))
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    U = np.concatenate(([Ux.reshape(-1)], [Uy.reshape(-1)],
                        [np.ones((xmax-xmin) * (ymax-ymin))]), axis=0)

    if direction == 'b':
        assert False
        # TODO: 3-1. apply H_inv to the destination pixels and retrieve (u,v) pixels
        H_inv = np.linalg.inv(H)
        V = H_inv @ U # np.dot(H_inv, U)
        V = V/V[-1]
        # TODO: 3-2. then reshape to (ymax-ymin),(xmax-xmin)
        Vx = V[0].reshape(ymax-ymin, xmax-xmin)
        Vy = V[1].reshape(ymax-ymin, xmax-xmin)
        # TODO: 4. calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (((Vx<w_src-1) & (0<=Vx)) & ((Vy<h_src-1) & (0<=Vy)))
        # TODO: 5-1. sample the source image with the masked
        mVx = Vx[mask]
        mVy = Vy[mask]
        # TODO: 5-2. interpolation and reshaped transformed coordinates
        mVxi = mVx.astype(int)
        mVyi = mVy.astype(int)
        p = interpolation(src, mVx, mVy,mVxi, mVyi)
        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax,xmin:xmax][mask] = p[mVyi,mVxi]

    elif direction == 'f':
        # TODO: 3-1. apply H to the source pixels and retrieve (u,v) pixels
        V = H @ U  # np.dot(H,U)
        V = (V/V[-1]).astype(int)
        
        # TODO: 3-2. then reshape to (ymax-ymin),(xmax-xmin)
        Vx = V[0].reshape(ymax-ymin, xmax-xmin)
        Vy = V[1].reshape(ymax-ymin, xmax-xmin)
        print(V.shape)
        # TODO: 4. calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = ((Vx<w_dst)&(0<=Vx))&((Vy<h_dst)&(0<=Vy))
        # TODO: 5. filter the valid coordinates using previous obtained mask
        mVx = Vx[mask]
        mVy = Vy[mask]
        # TODO: 6. assign to destination image using advanced array indicing
        dst[mVy, mVx, :] = src[mask]

    return dst
'''
def interpolation(src, mVx, mVy,mVxi, mVyi):
    dX = (mVx - mVxi).reshape((-1,1))
    dY = (mVy - mVyi).reshape((-1,1))
    p = np.zeros((src.shape))
    p[mVyi, mVxi, :] += (1-dY)*(1-dX)*src[mVyi, mVxi, :]
    p[mVyi, mVxi, :] += (dY)*(1-dX)*src[mVyi+1, mVxi, :]
    p[mVyi, mVxi, :] += (1-dY)*(dX)*src[mVyi, mVxi+1, :]
    p[mVyi, mVxi, :] += (dY)*(dX)*src[mVyi+1, mVxi+1, :]
    
    return p
'''