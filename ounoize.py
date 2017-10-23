from ddpg import ddpg_learner
import time

arr = []
for i in range(2000):
    n = ddpg_learner.OUNoise(1)
    print(n.noise())
    time.sleep(0.01)

