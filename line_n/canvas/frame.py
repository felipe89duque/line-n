import numpy as np

class SquareFrame:
    def __init__(self, width, height, mark_density):
        step = 1/mark_density
        x_vec = np.arange(0,width+step,step)
        y_vec = np.arange(0,height+step,step)
        top_edge = np.stack([x_vec,np.zeros(x_vec.size)],axis=1)
        right_edge = np.stack([width*np.ones(y_vec.size-1),y_vec[1:]],axis=1)
        bottom_edge = np.stack([x_vec[-2::-1],height*np.ones(x_vec.size-1)],axis=1)
        left_edge = np.stack([np.zeros(y_vec.size-2),y_vec[-2:0:-1]],axis=1)
        self.frame =np.concatenate([top_edge,right_edge,bottom_edge,left_edge],axis=0)
