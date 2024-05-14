from itertools import combinations
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

class Canvas:
    def __init__(self, frame_pts:np.ndarray,thread_width:float,thread_sharpness:float=1,thread_alpha=0.25,si_units=1e-2):
        self.frame_pts = frame_pts
        self.thread_width = thread_width
        self.thread_alpha=thread_alpha
        self.thread_sharpness=thread_sharpness
        self.tau = 1/thread_sharpness

        self.frame_px = (frame_pts/(thread_width*thread_sharpness)).astype(int)
        self.base = self.base_edges()

    @property
    def size(self):
        return self.frame_px.max(axis=0).astype(int) +1

    @property
    def edge_map(self):
        return ((edge_idx, point_pair) for edge_idx, point_pair in enumerate(combinations(self.frame_px,2)))

    def base_edges(self):
        # [K, H, W]
        base = {}
        for k,edge in self.edge_map:
            (x,y) = self.get_edge(*edge)
            base[k] = y,x
        
        return base

    def get_edge(self,point1:Tuple[int,int],point2:Tuple[int,int]) -> Tuple[np.ndarray,np.ndarray]:
        return self.bresenham(*point1,*point2)
        #return self.line(*point1,*point2)
        #return self.band(*point1,*point2)

    def edge2base(self,edge_idx):
        y_vec,x_vec = self.base[edge_idx]
        #image = -np.ones(self.size[[1,0]],dtype=float)
        image = np.zeros(self.size[[1,0]],dtype=float)
        image[y_vec,x_vec] = self.thread_alpha
        return image

    def edge2image(self,edge_idx:int) -> np.ndarray:
        y_vec,x_vec = self.base[edge_idx]
        image = np.zeros(self.size[[1,0]])
        image[y_vec,x_vec] = self.thread_alpha
        return image


    @staticmethod
    def bresenham(x0,y0,x1,y1):
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        
        x_vec = []
        y_vec = []
        while True:
            x_vec.append(x0)
            y_vec.append(y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * error
            if e2 >= dy:
                if x0 == x1:
                    break
                error = error + dy
                x0 = x0 + sx
            if e2 <= dx:
                if y0 == y1: 
                    break
                error = error + dx
                y0 = y0 + sy

        return (np.array(x_vec,dtype=int),np.array(y_vec,dtype=int))

    @staticmethod
    def line(x0,y0,x1,y1):
        dx = x1-x0
        dy = y1-y0
        num = abs(dx) + abs(dy)
        y = np.round(np.linspace(y0,y1,num))
        x = np.round(np.linspace(x0,x1,num))
        points = np.unique(np.stack([x,y],axis=1),axis=0)
        return (points[:,0].astype(int),points[:,1].astype(int))

  
    def plot(self,*edges,ax=None,**kwargs):
        if ax:
            plt.sca(ax)
        xs = self.frame_px[:,0]
        ys = self.frame_px[:,1]
        plt.scatter(xs,ys,**kwargs)
        
        all_edges = self.multiedge2image(*edges)
        plt.imshow(all_edges,cmap="binary",vmin=0,vmax=1)
        plt.xticks(xs,xs*self.thread_width)
        plt.yticks(ys,ys*self.thread_width)
        plt.gca().set_aspect("equal")

    def multiedge2image(self,*edges):
        if len(edges) == 0:
            return np.zeros(self.size[[1,0]])
        
        return np.sum([self.edge2image(edge) for edge in edges],axis=0).clip(min=0,max=1)
