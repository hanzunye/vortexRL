###
# this code test the speed of simulation in Vortex Studio
###
from cProfile import label
from Env_ import environment
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = environment()
env.reset()
a = np.array([[0.5,0.6,0.3]])
b = np.array([[-0.5,-0.6,-0.3]])

ratio1 = []
ratio2 = []
#env.render(False)
for _ in range(20):
    t_old = time.time()
    env.step_continuous(a)
    t_new = time.time()
    ratio1.append(1/(2*(t_new-t_old)))
    t_old = time.time()
    env.step_continuous(b)
    t_new = time.time()
    ratio1.append(1/(2*(t_new-t_old)))

env.reset()
for _ in range(20):
    t_old = time.time()
    env.step_continuous(a)
    t_new = time.time()
    ratio2.append(1/(2*(t_new-t_old)))
for _ in range(20):
    t_old = time.time()
    env.step_continuous(b)
    t_new = time.time()
    ratio2.append(1/(2*(t_new-t_old)))


fig, ax = plt.subplots()
ax.plot(0.5*np.arange(1, len(ratio1)+1), ratio1,label= 'without soil simulation')
ax.plot(0.5*np.arange(1, len(ratio2)+1), ratio2,label= 'with soil simulation')
ax.legend()
plt.ylabel('sim/real time ratio')
plt.xlabel('real time')
plt.savefig('ratiowithrender.png')    