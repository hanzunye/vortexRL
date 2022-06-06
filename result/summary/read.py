import csv
import matplotlib.pyplot as plt
import numpy as np

def scale(x):
    return -55+(x+55)*1.48

def trpo():
    exampleFile = open('log.csv')  # 打开csv文件
    exampleReader = csv.reader(exampleFile)  # 读取csv文件
    reward = []
    li = list(exampleReader)
    for i in range(len(li)):
        if i%2 == 0 and i!=0:
            reward.append(scale(float(li[i][0])))
    return reward


def ddpg():
    exampleFile = open('ddpg.csv')  # 打开csv文件
    exampleReader = csv.reader(exampleFile)  # 读取csv文件
    s = []
    reward = []
    li = list(exampleReader)

    li = [i for i in li[1:]]
    for a in li:
        s.append(float(a[1]))
    for i in range(50):
        reward.append(np.mean(s[20*i:(20*i+20)]))
    return reward

def ppo(): 
    exampleFile = open('ppo.csv')  # 打开csv文件
    exampleReader = csv.reader(exampleFile)  # 读取csv文件
    s = []
    reward = []
    li = list(exampleReader)

    li = [i for i in li[1:]]
    for a in li:
        s.append(float(a[1]))
    for i in range(50):
        reward.append(np.mean(s[20*i:(20*i+20)]))
    return reward

def reinforce():
    exampleFile = open('reinforce.csv')  # 打开csv文件
    exampleReader = csv.reader(exampleFile)  # 读取csv文件
    s = []
    reward = []
    li = list(exampleReader)

    li = [i for i in li[1:]]
    for a in li:
        s.append(float(a[1]))
    for i in range(50):
        reward.append(np.mean(s[20*i:(20*i+20)]))
    return reward

r1 = trpo()
r2 = ddpg()
r3 = ppo()
r4 = reinforce()
fig, ax = plt.subplots()
ax.plot(np.arange(1,51),r1,label = 'trpo')
ax.plot(np.arange(1,51),r2,label = 'ddpg')
ax.plot(np.arange(1,51),r3,label = 'ppo-discrete')
ax.plot(np.arange(1,51),r4,label = 'reinforce')

ax.legend()
plt.ylabel('Normalized reward')
plt.xlabel('Episode*20')
plt.title('Mean Normalized Rewards of each 20 episodes')
plt.savefig('result.png')  