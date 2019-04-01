import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class SideWalk:
    def __init__(self,l):
        self.length = l
        self.x = 0

    def observation(self):
        obs = np.zeros(self.length)
        obs[self.x] = 1
        return obs

    def reset(self):
        self.x = 0
        return self.observation()

    def step(self,action):
        done = False
        reward = -1
        if action == 0:
            self.x -= 1
        else:
            self.x += 1
        self.x = max(0,self.x)
        if self.x == length-1:
            done = True
            reward = 100
        return self.observation(),reward,done

class SideWalkWithObstacles:
    def __init__(self,l,w,scope=1,p=0.2):
        self.length = l
        self.width = w
        self.p = p
        self.x = int(l/2)
        self.y = int(w/2)
        self.scope = scope
        self.nstate = (2*scope+1)*(2*scope+1)
        self.obs = np.zeros((l,w))
        self.obs_ext = np.zeros((l+2*scope,w+2*scope))
        self.count = 0
        self.shown = False
        self.rects = None
        self.lastvisited = None

    def observation(self):
        return np.reshape(self.obs_ext[self.x:self.x+2*self.scope+1, \
                self.y:self.y+2*self.scope+1],self.nstate)
        #x = self.x+self.scope
        #y = self.y+self.scope
        #return np.array([self.obs_ext[x-1,y],self.obs_ext[x+1,y],self.obs_ext[x,y-1],self.obs_ext[x,y+1]])

    def reset(self):
        self.x = int(length/2)
        self.y = int(self.width/2)
        self.obs = np.random.rand(self.length,self.width) <= self.p
        self.obs[self.x,self.y] = False
        self.obs_ext = np.ones((self.length+self.scope*2, \
                self.width+self.scope*2))
        self.obs_ext[self.scope:-self.scope,self.scope:-self.scope] = \
                self.obs
        plt.close()
        self.count = 0
        self.shown = False
        self.fig = None
        self.rects = None
        self.lastvisited = None
        return self.observation()

    def step(self,action):
        self.count += 1
        done = self.count > 100
        reward = 1
        if action == 0:
            self.x -= 1
        elif action == 1:
            self.x += 1
        elif action == 2:
            self.y -= 1
        elif action == 3:
            self.y += 1
        if self.obs_ext[self.x+self.scope,self.y+self.scope]:
            self.x = max(0,min(self.x,self.length-1))
            self.y = max(0,min(self.y,self.width-1))
            reward = -100
            done = True
        return self.observation(),reward,done

    def render(self,bgColor='w',obstacleColor='gray'):
        if not self.shown:
            plt.show()
            fig = plt.figure()
            ax = fig.add_subplot(111) 
            rects = np.zeros((self.length,self.width), dtype = np.object)
            for i in np.arange(self.length):
                for j in np.arange(self.width):
                    rect = Rectangle((i,j),1,1)
                    ax.add_artist(rect)
                    if self.obs[i,j]:
                        rect.set_facecolor(obstacleColor)
                    else:
                        rect.set_facecolor(bgColor)
                    rect.set_edgecolor('k')
                    rects[i,j] = rect
            ax.set_xlim(0,self.length)
            ax.set_ylim(0,self.width)
            ax.set_aspect('equal','box')
            self.rects = rects
            self.fig = fig
            self.shown = True
        if self.lastvisited != None:
            self.rects[self.lastvisited[0],self.lastvisited[1]]. \
                    set_facecolor('b' if self.obs[self.lastvisited[0],self.lastvisited[1]] else 'g')
        self.rects[self.x,self.y].set_facecolor('r')
        self.lastvisited = (self.x,self.y)
        plt.pause(0.1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

render = False 
# hyperparameters
H = 20 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False

# model initialization
D1 = 9 # input dimensionality: 80x80 grid
D2 = 4
model = {}
model['W1'] = np.random.randn(H,D1) / np.sqrt(D1) # "Xavier" initialization
model['W2'] = np.random.randn(D2,H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0,r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp, epx):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(epdlogp.T,eph)
  dh = np.dot(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}
  
def select(p):
    p = np.cumsum(p)
    r = np.random.rand()
    for i in np.arange(p.size):
        if r < p[i]:
            return i

ti = time.process_time()
length = 20
width = 20 
env = SideWalkWithObstacles(length,width,scope=1)
observation = env.reset()
xs,hs,dlogps,drs = [],[],[],[]
episode_number = 0
steps = 0
sumsteps = 0
old_sumsteps = 0
max_episodes = 10004
while episode_number < max_episodes:
    if render: env.render()
    if episode_number > max_episodes : env.render()
    steps += 1 

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(observation)
    aprob = np.array(aprob) / np.sum(aprob)
    action = select(aprob)

    # record various intermediates (needed later for backprop)
    xs.append(observation) # observation
    hs.append(h) # hidden state
    y = np.zeros(4)
    y[action] = 1
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done = env.step(action)
    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr = discounted_epr - np.mean(discounted_epr)
        if discounted_epr.size > 1:
            discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp, epx)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        sumsteps += steps
        if episode_number % 1000 == 0: 
            print('ep %d: sidewalk crossed, steps: %d, mean steps: %f' % (episode_number,steps,(sumsteps-old_sumsteps)/1000))
            old_sumsteps = sumsteps
        observation = env.reset()
        steps = 0

print('time elapsed: %f s' % (time.process_time() - ti))
