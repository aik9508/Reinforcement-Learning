import numpy as np
import approach
import obstacle
import display
import matplotlib.pyplot as plt

length = 25
width = 3
epsilon = 0.9
naction = 4
w_approach = 0.4
w_obstacle = 1 - w_approach
Qapproach = approach.train(length,width)
Qobstacle = obstacle.train(20,20)
_,_,Qnorm_approach = approach.strategy(Qapproach)
_,_,Qnorm_obstacle = obstacle.strategy(Qobstacle)
obstacles = np.random.rand(length,width)<0.2
S = obstacle.computeStates(obstacles)

x = 0
y = np.random.randint(0,width)
episode = np.array([x,y])
s_approach = x
s_obstacle = S[x,y]

while x < length - 1:
    q = Qnorm_approach[s_approach,:] * w_approach + \
            Qnorm_obstacle[s_obstacle,:] * w_obstacle
    if np.random.rand() < epsilon:
        a = np.argmax(q)
    else:
        a = np.random.randint(naction)
    x,y = approach.updateState(x,y,a)
    x = max(0,min(length-1,x))
    y = max(0,min(width-1,y))
    s_approach = x
    s_obstacle = S[x,y]
    episode = np.append(episode,np.array([x,y]))

xhistory = episode[0:-2:2]
yhistory = episode[1:-2:2]
fig,_,rects=display.drawSidewalk(obstacles)
plt.pause(1)
rects[xhistory[0],yhistory[0]].set_facecolor('r')
plt.pause(0.1)
for i in np.arange(1,xhistory.size):
    if obstacles[xhistory[i-1],yhistory[i-1]]:
        rects[xhistory[i-1],yhistory[i-1]].set_facecolor('b')
    else:
        rects[xhistory[i-1],yhistory[i-1]].set_facecolor('g')
    rects[xhistory[i],yhistory[i]].set_facecolor('r')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)






