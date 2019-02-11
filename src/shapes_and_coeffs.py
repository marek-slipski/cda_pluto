import numpy as np
import scipy, scipy.ndimage, scipy.interpolate
import cv2
import sys

def remove_edge_shapes(img,conts):
    check_x = [0,img.shape[0]-1]
    check_y = [0,img.shape[1]-1]
    cons_keep = [] #initialize indicies to keep
    for k,eachc in enumerate(conts): # go through each contour
        c_rs = eachc.reshape(len(eachc),2) #reshape to get x,y
        x_bs, y_bs = c_rs.T #get x and y pixels separately
        if any(cx in x_bs for cx in check_x): #if x or y on edge of frame, ignore
            continue
        elif any(cy in y_bs for cy in check_y):
            continue
        else:
            cons_keep.append(k) #otherwise, keep it
    return cons_keep


def centroid(cont):
    ts_m = cv2.moments(cont) #get info on boundary
    ts_cx = int(ts_m['m10']/ts_m['m00']) #centroid x
    ts_cy = int(ts_m['m01']/ts_m['m00']) #centroid y
    return ts_cx, ts_cy

def theta_arctan(x,y):
    if x > 0:
        temp = np.arctan(float(y)/x) #base
        if y >= 0:
            return temp #between 0 and pi/2
        elif y < 0:
            return temp + 2*np.pi #from -pi/2 to 0 --> 3pi/2 to 2pi
    elif x < 0:
        return np.arctan(float(y)/x) + np.pi
    else:
        if y > 0:
            return np.pi/2
        else:
            return 3*np.pi/2

def dist_theta(cont):
    dist = np.zeros((len(cont))) #init distance
    theta = np.zeros((len(cont))) #init thetas
    ts_cx, ts_cy = centroid(cont)
    for i,(x,y) in enumerate(cont):
        xdist = (x-ts_cx)
        ydist = (y-ts_cy)
        dist[i] = np.sqrt(xdist**2+ydist**2)
        theta[i] = theta_arctan(xdist,ydist)
    return dist,theta

def coeffs(x,y):
    ########################################################################
    # GETTING FOURIER COEFFICIENTS
    # TAKEN FROM https://stackoverflow.com/questions/13604611/how-to-fit-a-closed-contour
    comp =  x + y *1.j #convert x,y to complex plane
    thetas = np.angle(comp) #find the angle to the point
    distances = np.absolute(comp) # and the distance to the point
    sortidx = np.argsort( thetas ) #organize by angle using indicies
    thetas = thetas[ sortidx ]
    distances = distances[ sortidx ]

    # copy first and last elements with angles wrapped around. needed so can interpolate over full range -pi to pi
    thetas = np.hstack(([ thetas[-1] - 2*np.pi ], thetas, [ thetas[0] + 2*np.pi ]))
    distances = np.hstack(([distances[-1]], distances, [distances[0]]))

    # interpolate to evenly spaced angles
    f = scipy.interpolate.interp1d(thetas, distances)
    thetas_uniform = scipy.linspace(-np.pi, np.pi, num=100, endpoint=False)
    distances_uniform = f(thetas_uniform)

        # fft and inverse fft
    fft_coeffs = np.fft.rfft(distances_uniform)
    return fft_coeffs


if __name__=='__main__':
    import matplotlib
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt

    import os

    deps = np.load(sys.argv[1]) # load binary image
    save_dir = sys.argv[2]
    save_shape = save_dir+'/shapes'
    os.system('mkdir -p '+save_shape) #make directory for files

    # get contours
    im2,contours,hier = cv2.findContours(np.copy(deps).astype('uint8'), \
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # plot all detected contours
    plt.figure()
    for i in range(len(contours)):
        plt.imshow(cv2.drawContours(im2,contours,i,100,thickness=3),cmap='Greys')
    plt.savefig(save_shape+'/'+'contours_all.png',dpi=300)
    #plt.show()

    con_inds = remove_edge_shapes(deps,contours) #remove things on the edge
    cont_adj = [contours[z].reshape(len(contours[z]),2) for z in con_inds]

    # plot only those contours in fully in image
    plt.figure()
    for j in con_inds:
        plt.imshow(cv2.drawContours(deps,contours,j,100,thickness=3),cmap='Greys')
    plt.savefig(save_shape+'/'+'contours_adj.png',dpi=300)
    #plt.show()

    #initialize figure, angles
    rec_fig, rec_ax = plt.subplots()
    rec_ax.imshow(deps)
    arb_ang = scipy.linspace(-np.pi, np.pi, num=100, endpoint=False)
    for t,cont in enumerate(cont_adj):
        cent_x,cent_y = centroid(cont)
        x,y = cont.T
        xdiff = x - cent_x
        ydiff = y - cent_y
        np.save(save_shape+'/'+str(t)+'_xypoints',np.array([x,y]))

        # Get Fourier Coeffs
        f_c = coeffs(xdiff,ydiff)
        r_fit = np.fft.irfft(f_c)

        #dists,thetas = dist_theta(cont)
        #to_save = np.array([[cent_x,cent_y],dists,thetas]) #save centroid,r,theta
        #np.save(save_shape+'/'+str(t)+'_points',to_save)

        # check radial and thetas
        #rx = cent_x + dists*np.cos(thetas)
        #ry = cent_y + dists*np.sin(thetas)

        #plot fourier coefficient amplitudes
        An = np.sqrt(f_c.real**2+f_c.imag**2) #amplitudes
        plt.figure()
        plt.plot(np.arange(len(An)),An)
        plt.xlim(1,10)
        plt.ylim(0,An[1:].max()+5)
        plt.savefig(save_shape+'/'+str(t)+'_CoeffAmps')

        #check points
        rec_ax.plot(cent_x,cent_y,'ro',ms=3)
        rec_ax.plot(r_fit*np.cos(arb_ang)+cent_x,r_fit*np.sin(arb_ang)+cent_y,'k',lw=3)
    rec_fig.savefig(save_shape+'/PointsExtracted.png',dpi=300)
    plt.show()
