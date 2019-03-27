import numpy as np
import matplotlib.pyplot as plt
import display

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
naction = 4

def reward(s,a):
    if a == RIGHT:
        return s 
    elif a == LEFT:
        return -50 
    else:
        return -100

def updateState(x,y,a):
    if a == RIGHT:
        x = x + 1
    elif a == LEFT:
        x = x - 1
    elif a == UP:
        y = y - 1
    else:
        y = y + 1
    return x,y 

def randomSelect(pSum):
    r = np.random.rand()
    for a in np.arange(pSum.size):
        if r < pSum[a]:
            return a

def strategy(Q):
    m = np.mean(Q,axis=1)
    p = Q - np.repeat(m[:,np.newaxis],Q.shape[1],axis=1)
    p = 1/(1+np.exp(-p))
    Qnorm = p/np.max(p)
    b = np.sum(p,axis=1)
    p = p/np.repeat(b[:,np.newaxis],Q.shape[1],axis=1)
    pSum = np.cumsum(p,axis=1)
    return p,pSum,Qnorm

def train(length,width,gamma=0.9,epsilon=0.9,alpha=0.1,max_iter=1000):
    # parameters
    nstate = length
    Q = np.random.rand(nstate,naction)
    R = np.zeros((nstate,naction))

    for s in np.arange(nstate):
        for a in np.arange(naction):
            R[s,a] = reward(s,a)

    for i in np.arange(1000):
        x = 0
        y = np.random.randint(0,width)
        s = x

        for it in np.arange(max_iter):
            a = np.argmax(Q[s,:])
            if np.random.rand() > epsilon:
                a = np.random.randint(0,naction)
            # take action a, observe s_{t+1}
            x,y = updateState(x,y,a)
            x = max(0,min(x,length-1))
            y = max(0,min(y,width-1))
            snext = x
            Q[s,a] = Q[s,a] + alpha * (R[s,a] + gamma * np.max(Q[snext,:]) \
                    - Q[s,a])
            s = snext
            if x >= length - 1:
                break
    return Q

def main():
    length = 25
    width = 3
    epsilon = 0.9
    Q = train(length,width)
    x = 0
    y = np.random.randint(0,width)
    episode = np.array([x,y])
    s = x
    while x < length - 1:
        #a = randomSelect(pSum[s,:])
        a = np.argmax(Q[s,:])
        if np.random.rand() > epsilon:
            a = np.random.randint(0,naction)
        x,y = updateState(x,y,a)
        x = max(0,min(x,length-1))
        y = max(0,min(y,width-1))
        s = x
        episode = np.append(episode,np.array([x,y]))

    xhistory = episode[0:-2:2]
    yhistory = episode[1:-2:2]
    obstacles = np.zeros((length,width),dtype=bool)
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
        plt.pause(0.05)
    print episode[0::2]
    print episode[1::2]
    print Q

# main()

