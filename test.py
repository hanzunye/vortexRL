from Env_ import environment
import numpy as np
env = environment()
env.reset()
a = np.array([[0.5,0.6,0.3]])
b = np.array([[-0.5,-0.6,-0.3]])
for _ in range(10):
    env.step_continuous(a)
    #env.step_continuous(b)

for _ in range(10):
    env.step_continuous(a)
    #env.step_continuous(b)
