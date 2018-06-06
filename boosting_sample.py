import numpy as np
import random
from machine import *

# number of elements
s = 20

x = []
y = []
w = s*[1.0/s]
weak_learners = []
learner_used = {}
random.seed()
# s random points for boosting
for i in range(s):
    pom = random.sample([1,-1], 1)
    print pom[0]
    y.append(pom[0])
    xx = random.randrange(50,750,6)
    yy = random.randrange(50, 750,20)
    x.append([xx,yy])
    if not xx in learner_used:
        learner_used[xx] = 1
        weak_learners.append(xx+5)
        weak_learners.append(-xx-5)


boost = Machine(x,y,w,weak_learners)
boost.learn(s*2)


for i in range(10):
    xx = random.randrange(50, 750, 6)
    yy = random.randrange(50, 750, 20)
    point = [xx, yy]
    boost.classify(point)


boost.canvas.bind("<Button-1>", boost.classify_gui_point)
boost.canvas.pack()

boost.root.mainloop()
