import numpy as np
import approach
import obstacle
import display
import matplotlib.pyplot as plt

def select(q):
    qmax = np.max(q)
    #qmin = min(0,np.min(q))
    qmin = 0
    equivalent_actions = np.where((qmax - q) < 0.05 * (qmax - qmin))[0]
    #print q,equivalent_actions.size
    return equivalent_actions[np.random.randint(0,equivalent_actions.size)]

def norm(Q):
    shape = Q.shape
    Qnorm = np.zeros(shape)
    for i in np.arange(shape[0]):
        qmax = np.max(Q[i,:])
        #qmin = min(0,np.min(Q[i,:]))
        #Qnorm[i,:] = (Q[i,:] - qmin)/(qmax - qmin + 1e-16)
        q = np.maximum(Q[i,:],0)
        Qnorm[i,:] = q/(qmax + 1e-16)
    return Qnorm

length = 25
width = 7 
epsilon = 0.95
naction = 4
w_approach = 0.4
w_obstacle = 1 - w_approach
print 'approach module'
Qapproach = approach.train(length,width)
print 'obstacle avoidance module'
Qobstacle = obstacle.train(10,10)
print 'finish training'
Qnorm_approach = norm(Qapproach)
Qnorm_obstacle = norm(Qobstacle)
obstacles = np.random.rand(length,width)<0.2
S = obstacle.computeStates(obstacles)
print Qnorm_approach
print Qnorm_obstacle

x = 0
y = np.random.randint(0,width)
episode = np.array([x,y])
s_approach = x
s_obstacle = S[x,y]

while x < length - 1:
    q = Qnorm_approach[s_approach,:] * w_approach + \
            Qnorm_obstacle[s_obstacle,:] * w_obstacle
    if np.random.rand() < epsilon:
        #print Qnorm_approach[s_approach,:], Qnorm_obstacle[s_obstacle,:],
        a = select(q)
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



