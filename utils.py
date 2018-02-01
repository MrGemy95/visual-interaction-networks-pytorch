import os
import matplotlib.pyplot as plt
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def make_image2(xy,img_folder,prefix):
  if not os.path.exists(img_folder):
    os.makedirs(img_folder);
  fig_num=len(xy);
  mydpi=100;
  for i in range(fig_num):
    #fig = plt.figure(figsize=(32/mydpi,32/mydpi))
    fig = plt.figure(figsize=(128/mydpi,128/mydpi))
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.axis('off');
    color=['r','b','g','k','y','m','c'];
    for j in range(len(xy[0])):
      #plt.scatter(xy[i,j,1],xy[i,j,0],c=color[j%len(color)],s=0.5);
      plt.scatter(xy[i,j,1],xy[i,j,0],c=color[j%len(color)],s=5);
    fig.savefig(img_folder+prefix+"_"+str(i)+".png",dpi=mydpi);