import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def drawSidewalk(obstacles,bgColor='w',obstacleColor='gray'):
    m = obstacles.shape[0]
    n = obstacles.shape[1]
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    rects = np.zeros((m,n), dtype = np.object)
    for i in np.arange(m):
        for j in np.arange(n):
            rect = Rectangle((i,j),1,1)
            ax.add_artist(rect)
            if obstacles[i,j]:
                rect.set_facecolor(obstacleColor)
            else:
                rect.set_facecolor(bgColor)
            rect.set_edgecolor('k')
            rects[i,j] = rect
    ax.set_xlim(0,m)
    ax.set_ylim(0,n)
    ax.set_aspect('equal','box')
    return fig,ax,rects

#length = 25
#width = 3
#obstacles = np.random.rand(length,width) <= 0.2
#fig,ax,rects = drawSidewalk(obstacles)
#plt.pause(1)
#rects[1,2].set_facecolor('r')
#fig.canvas.draw()
#fig.canvas.flush_events()
#plt.pause(1)


