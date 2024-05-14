import numpy as np

from line_n import Canvas


class Projector:   
   def __init__(self, canvas:Canvas,image:np.ndarray,transform=None,transform_kwargs=None):
      self.canvas=canvas
      #self.image = 2*(image.astype(float)/255 - 0.5)
      self.image = image.astype(float)/255
      self.stop = False
      self.metric = -np.inf
      self.steps=0
      self.max_steps = 1000
      self.metrics =[]
      self.transform = transform
      self.transform_kwargs = transform_kwargs

   @property
   def pruned_base(self):
      return [edge for edge in self.prune_edges()]
      
   @property
   def max_projection(self):
      return np.product(self.image.shape)
   

   def get_spectrum(self,image):
      dot = np.empty(len(self.canvas.base.keys()),dtype=float)
      for edge_idx, _ in self.canvas.edge_map:
         edge_img = self.canvas.edge2base(edge_idx)
         dot[edge_idx] = np.sum(image*edge_img) 
      return dot

   def project(self,image, transform,kwargs):
      trans_input_img = transform(image,**kwargs)
      trans_self_img = transform(self.image,**kwargs)
      #plt.figure()
      #plt.imshow(trans_input_img*trans_self_img)
      #plt.colorbar()
      return np.sum(trans_input_img*trans_self_img)
      
   def prune_edges(self):
      new_img = self.image.copy()
      spectrum = self.get_spectrum(self.image)
      self.max_dots = []
      max_dot = spectrum.argmax()
      similarity = -np.inf

      #while (new_img.clip(min=0).sum() > self.image.sum()*self.energy_threshold) and (spectrum[max_dot] >0):
      while not self.stop:
         self.max_dots.append(max_dot)
         yield max_dot
         new_img -= self.canvas.edge2image(max_dot) * self.canvas.thread_alpha

         spectrum = self.get_spectrum(new_img)
         max_dot = spectrum.argmax()
         self.optimiser_step()


         #print(new_img.clip(min=0).sum(),self.image.sum(),self.image.sum()*self.energy_threshold,spectrum[max_dot])

      return None

   def optimiser_step(self):
      image = self.canvas.multiedge2image(*self.max_dots)
      metric = self.project(image,self.transform,self.transform_kwargs)
      self.metrics.append(metric)
      metric_diff = metric - self.metric
      self.metric = metric
      if metric_diff <=0:
         self.stop = True
      
      elif self.steps >= self.max_steps:
         print("Max steps reached")
         self.stop = True

      self.steps +=1

   

   def draw(self,ax=None,**kwargs):
      ax = ax or plt.axes()

      edges=self.prune_edges()
      projection = np.zeros(self.canvas.size[[1,0]])
      for edge in edges:
         projection += self.canvas.edge2image(edge)*self.canvas.thread_alpha
      projection= (projection.clip(max=1)*255//projection.max()).astype(np.uint8)
      ax.imshow(projection,**kwargs)
