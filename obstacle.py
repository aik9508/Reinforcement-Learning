import numpy as np
import matplotlib.pyplot as plt
import display

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
naction = 4

def reward(s,a):
    if s >> a & 1:
        return -100;
    return 1

def state(x,y,obstacles_extend):
    s = 0
    neighbors = np.array([obstacles_extend[x,y-1], \
            obstacles_extend[x,y+1], \
            obstacles_extend[x-1,y], \
            obstacles_extend[x+1,y], \
            obstacles_extend[x-1,y-1], \
            obstacles_extend[x-1,y+1], \
            obstacles_extend[x+1,y-1], \
            obstacles_extend[x+1,y+1]])
    for i in np.arange(neighbors.size):
        s = s + (neighbors[i]<<i)
    return s

def computeStates(obstacles):
    obstacles_extend = np.ones((obstacles.shape[0]+2, \
            obstacles.shape[1]+2),dtype=bool)
    obstacles_extend[1:-1,1:-1]=obstacles
    S = np.zeros(obstacles.shape,dtype=int)
    for x in np.arange(obstacles.shape[0]):
        for y in np.arange(obstacles.shape[1]):
            S[x,y] = state(x+1,y+1,obstacles_extend)
    return S

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

def select(q):
    qmax = np.max(q)
    #qmin = min(0,np.min(q))
    qmin = 0
    equivalent_actions = np.where((qmax - q) <= 0.05 * (qmax - qmin))[0]
    if equivalent_actions.size == 0:
        return np.argmax(q)
    return equivalent_actions[np.random.randint(0,equivalent_actions.size)]

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

#Q = np.zeros((nstate,naction))
def train(length,width,gamma=0.99,max_iter=100):
    epsilon = 0
    # parameters
    nstate = 256
    Q = np.zeros((nstate,naction))
    #Q = np.random.rand(nstate,naction)
    R = np.zeros((nstate,naction))
    C = np.zeros((nstate,naction))

    for s in np.arange(nstate):
        for a in np.arange(naction):
            R[s,a] = reward(s,a)

    for i in np.arange(10000):
        if epsilon < 0.95 and i%100==0:
            epsilon = epsilon + 0.01
        Qold = np.copy(Q)
        obstacles = np.random.rand(length,width) <= 0.2
        S = computeStates(obstacles)
        x = np.random.randint(0,length)
        y = np.random.randint(0,width)
        s = S[x,y]
        
        for it in np.arange(max_iter):
            if np.random.rand() > epsilon:
                a = np.random.randint(0,naction)
                #a = randomSelect(Q[s,:])
                #if s == 0:
                #    print a
            else:
                a = select(Q[s,:])
            C[s,a] = C[s,a] + 1
            # take action a, observe s_{t+1}
            x,y = updateState(x,y,a)
            if x < 0 or x >= length or y < 0 or y >= width:
                break
            else:
                snext = S[x,y]
            alpha = 1/C[s,a]
            if R[s,a] > 0:
                Q[s,a] = Q[s,a] + alpha * (R[s,a] + gamma * np.max(Q[snext,:]) \
                        - Q[s,a])
            else:
                Q[s,a] = Q[s,a] + alpha * (R[s,a] - Q[s,a])
                break
            s = snext
        #if np.linalg.norm(Q-Qold) < 1e-3:
        #    break
    print np.linalg.norm(Q-Qold)
    return Q

def main():
    length = 20
    width = 20
    Q = train(length,width,gamma=0.9,max_iter=1000)
    p,pSum,_ = strategy(Q)
    obstacles = np.random.rand(length,width) <= 0.2
    S = computeStates(obstacles)
    # test
    x = int(length/2)
    y = int(width/2)
    episode = np.array([x,y])
    s = S[x,y]
    for it in np.arange(100):
        #a = randomSelect(pSum[s,:])
        if np.random.rand() < 0.95:
            a = select(Q[s])
        else:
            a = np.random.randint(0,naction)
        x,y = updateState(x,y,a)
        if x < 0 or x >= length or y < 0 or y >= width:
            break
        else:
            s = S[x,y]
        episode = np.append(episode,np.array([[x,y]]))

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
        plt.pause(0.05)
    with open('q.txt','w') as f:
        for j in np.arange(Q.shape[0]):
            for k in np.arange(Q.shape[1]):
                f.write("%f\t" % Q[j,k])
            f.write("\n")
    print p[0:32,:]
    print Q[0:32,:]

#main()
#print episode[0::2]
#print episode[1::2]
#print Q
