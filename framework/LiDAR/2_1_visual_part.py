import matplotlib.pyplot as plt
import os, sys, glob
from natsort import natsorted
import numpy as np
import cv2
import math
import pandas as pd
from PIL import Image
from PIL import ImageOps
"""
Calculate SMI Metric behavior with steepest gradient descent optimization one each lidar (XYZr)data in KITTI folder
LiDAR feature - reflectivity
Image feature - grayscale intensity
"""
pi2 = math.pi/2
def print_help_and_exit():
    print('Usage: .py [from KITTI bin file dir]')
    sys.exit()
    
def calc_val(line):
    line = line.strip("[]\n")
    line = line.split(',', 3)
    X = float(line[0])
    Y = float(line[1])
    Z = float(line[2])
    inten = int(line[3])
    return ([X, Y, Z, inten])

def read_pcap(lidar_file):
    pcap_data = []
    f = open(lidar_file, "r")
    line_counter = 0
    for line in f:
        pcap_data.append(calc_val(line))
        line_counter += 1
    return (pcap_data)

def read_img(data):
    img = Image.open(data)
    return (img)

def read_calib_data(calib_dir):
    cam_mono = cv2.FileStorage(calib_dir + "/cam_mono.yml", cv2.FILE_STORAGE_READ)
    K = cam_mono.getNode("K")
    return (K.mat())

def get_intensivity(pixel, img):
    x = int(pixel[0] + 959)
    y = int(pixel[1] + 539)
    return img[y][x]


def Projection2(Lidar_data, points, K):
    """
    (x, y, z) - scanned point in LiDAR coordinates
    (u, v, w) - corresponding point  in camera coordinates
    (u0, v0, w0, 1) - the translation vector (u0=0.885,v0=0, w0=1.535)  
    R = Rroll*Rpitch*Ryaw -> 3*3 matrixes with the combination of cos and sin (alpha0=0,betta0=0,gamma0=0)
    points = [alpha, beta, gamma, u0, v0, w0]
    Lidar_data[0:3] = x y z values
    |u|    |R     t| |x|
    |v|  = |       | |y|
    |w|    |       | |z| 
    |-|    |0     1| |1|
    (i, j) - return corresponding pint in the image plane (pinhole model):
    (i0, j0)read from K matrix from calibration file info
    |i|    |fx/w   0    i0| |u|
    |j|  = | 0    fy/w  j0| |v|
    |0|    |              | |1| 
    """
    roll = [[1, 0, 0], [0, np.cos(points[0]), -np.sin(points[0])], [0, np.sin(points[0]), np.cos(points[0])]]
    pitch = [[np.cos(points[1]), 0, np.sin(points[1])], [0, 1, 0], [-np.sin(points[1]), 0, np.cos(points[1])]]
    yaw = [[np.cos(points[2]), -np.sin(points[2]), 0], [np.sin(points[2]), np.cos(points[2]), 0], [0, 0, 1]]
    R = np.dot(roll, np.dot(pitch, yaw)) 
    point_coord=Lidar_data[:3]
    #print ("data", point_coord)
    rotation = np.dot(R, point_coord)
    #print ("rotation",rotation)
    coord=[rotation[0]+points[3],rotation[1]+points[4],rotation[2]+points[5]]
    #print ("coor",coord)
    w = coord[2]
    P = np.zeros([3, 3])
    P[:, :] = K[:, :]
    P[0][0] = P[0][0]/ w
    P[1][1] = P[1][1]/ w
    pixels = np.dot(P, coord)
    if (abs(pixels[0]) < 960) and (abs(pixels[1]) < 540):
        d=np.sqrt(sum(i*i for i in coord))
        pixels[2] = d
        #print ("pixels",pixels)
        return (pixels)
    
    
def Projection(Lidar_data, points, K):
    """
    (x, y, z) - scanned point in LiDAR coordinates
    (u, v, w) - corresponding point  in camera coordinates
    (u0, v0, w0, 1) - the translation vector (u0=0.885,v0=0, w0=1.535)  
    R = Rroll*Rpitch*Ryaw -> 3*3 matrixes with the combination of cos and sin (alpha0=0,betta0=0,gamma0=0)
    points = [alpha, beta, gamma, u0, v0, w0]
    Lidar_data[0:3] = x y z values
    |u|    |R     t| |x|
    |v|  = |       | |y|
    |w|    |       | |z| 
    |-|    |0     1| |1|
    (i, j) - return corresponding pint in the image plane (pinhole model):
    (i0, j0)read from K matrix from calibration file info
    |i|    |fx/w   0    i0| |u|
    |j|  = | 0    fy/w  j0| |v|
    |0|    |              | |1| 
   
    roll = [[1, 0, 0], [0, np.cos(points[0]), -np.sin(points[0])], [0, np.sin(points[0]), np.cos(points[0])]]
    pitch = [[np.cos(points[1]), 0, np.sin(points[1])], [0, 1, 0], [-np.sin(points[1]), 0, np.cos(points[1])]]
    yaw = [[np.cos(points[2]), -np.sin(points[2]), 0], [np.sin(points[2]), np.cos(points[2]), 0], [0, 0, 1]]
    R = np.dot(roll, np.dot(pitch, yaw)) 
    position = [Lidar_data[0]+points[3], Lidar_data[1]+points[4], Lidar_data[2]+points[5]]
    coord = np.dot(R, position) #(u, v, w)
    w = coord[2]
    P = np.zeros([3, 3])
    P[:, :] = K[:, :]
    P[0][0] = P[0][0]/ w
    P[1][1] = P[1][1]/ w
    pixels = np.dot(P, coord)
    if (abs(pixels[0]) < 960) and (abs(pixels[1]) < 540):
        d=np.sqrt(sum(i*i for i in coord))
        pixels[2] = d
        
        return (pixels)
     """
   
def dfScatter(img, df, xcol='x', ycol='y', catcol='z'):
    fig, ax = plt.subplots(figsize=(20, 10), dpi=60,)
    categories = np.unique(df[catcol])
    colors = np.linspace(categories.min(), categories.max(), len(categories))
    colordict = dict(zip(categories, colors))  
    df["c"] = df[catcol].apply(lambda k: colordict[k])
    img = ImageOps.mirror(img)
    
    sc= plt.scatter(df[xcol],df[ycol] , c=df.c ,zorder=2, s=10)
    #plt.imshow(img,extent=[df[xcol].min(),df[xcol].max(),df[ycol].min(),df[ycol].max()], zorder=0, aspect='auto')
    colorize=plt.colorbar(sc, orientation="horizontal")
    colorize.set_label("distance (m)")
    return fig

def read_data(folder):
    Video_Flows = []
    for f in sorted(os.listdir(folder)):
        f_name = os.path.split(os.path.splitext(f)[0])[-1]
        Video_Flows.append(f_name)
    for Video_Flow in Video_Flows:
        Lidar_path = os.path.join(folder, Video_Flow, 'velodyne_points', 'data')
        pcap_files = natsorted(glob.glob(Lidar_path + '/*.txt'))
        Lidar_data = [read_pcap(i) for i in pcap_files]
        #df_Lidar_data = pd.DataFrame(Lidar_data)) 
        Camera_path = os.path.join(folder, Video_Flow, 'leftImage', 'data')
        images_files = natsorted(glob.glob(Camera_path + '/*.bmp'))
        images = [read_img(i) for i in images_files]
      
        K = np.array(read_calib_data(os.path.join(folder, Video_Flow, 'calib')))
        """
        cam=[-pi2,0 , 2*pi2, -0.885, -0.066, 0]
        points = [1, 2, 3]
        pixel = Projection2(points, cam, K)
        """
        
        camera_coord2 = [-pi2,0 , 2*pi2, -0.885, -0.066, 0]
        
        for i in range(25):
            dataset = Lidar_data[i]
            pixels = []
            for j in range(len(Lidar_data[i])):
                pixel = Projection2(dataset[j][:3], camera_coord2, K)
                if pixel is not None:
                    pixels.append(pixel)
            df = pd.DataFrame(pixels, columns = ['x' , 'y', 'z']) 
            fig = dfScatter(images[i],df)
            fig.savefig(str(i)+'.png', dpi =60)
            
        
        
        
        
        '''
        #camera_coord = [-pi2, 0, 2*pi2, 0.885, 0, 0.066]
        camera_coord = [pi2, 0, 2*pi2 - pi2/18, 0.885, 0, 0.066]
        
        for i in range(40):
            dataset = Lidar_data[i]
            pixels = []
            for j in range(len(Lidar_data[i])):
                data = dataset[j][:3]
                pixel = Projection(data, camera_coord, K)
                if pixel is not None:
                    pixels.append(pixel)
            df = pd.DataFrame(pixels, columns = ['x' , 'y', 'z']) 
            fig = dfScatter(images[i], df)
            
            strFile = str(i)+'.png'
            fig.savefig(strFile, dpi =60)
            plt.close()
            
        
        plt.scatter(pixels[:,0], pixels[:,1])
        plt.ylabel('Pixels')
        plt.show() 
        
       
        #average_points = [0, 0, 0, 0.885, 0, -1.535]
        #cur_points = [0, 0, 0, 0.885, 0, -1.535]
        #delta = 0.000001
        # POINT IN (00)
        camera_coord = [-pi2, 0, 2*pi2, 0.885, 0, -0.066]
        print(Projection([-0.885,1,0.066],camera_coord,K))
        
        plt.subplot(311)
        plt.plot(x)
        plt.title('Lidar is watching you')
        plt.ylabel('x')
        
        plt.subplot(312)
        plt.plot(y)
        plt.ylabel('y')
        
        plt.subplot(313)
        plt.plot(z)
        plt.ylabel('z')
        
        plt.show()
        '''
def main():
    if len(sys.argv) == 2:
        print ('unpack from', sys.argv[1])
        read_data(sys.argv[1])
    else:
        print_help_and_exit()

if __name__ == '__main__':
    main()