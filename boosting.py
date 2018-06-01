import math
import numpy as np


class Machine:
    def __init__(self, samples, desired_outputs, init_weights, weak_learners):
        self.x = samples
        self.y = desired_outputs
        self.w = init_weights
        self.weak_learners = list(weak_learners)
        self.n = len(desired_outputs)
        self.correctly_classified = [True]*self.n
        self.z = None
        self.e = None
        self.a = None
        self.alphas = []
        self.errors = [None]*len(weak_learners)

    def pick_h(self):
        min_err = 1000
        min_err_pos = -1
        for i in range(len(self.weak_learners)):
            self.errors[i] = self.error(self.weak_learners[i])
            '''
            # print "h:", i, err
            print "old error",self.errors[i],
            if self.errors[i] > 0.5:
                self.weak_learners[i] *= -1
            print "new error", self.error(self.weak_learners[i])

            self.errors[i] = self.error(self.weak_learners[i])
            '''
            if self.errors[i] < min_err:
                min_err = self.errors[i]
                min_err_pos = i

        print "min error ",min_err
        return self.weak_learners[min_err_pos], min_err_pos

    def error(self, h):
        err = 0
        for i in range(len(x)):
            if self.dif(h, x[i]) != self.y[i]:
                err += self.w[i]
            '''
            if self.x[i][0] < h and not self.y[i] == -1:
                # error, wrong side of the line
                err += self.w[i]
            if self.x[i][0] >= h and not self.y[i] == 1:
                # error, wrong side of the line
                err += self.w[i]
            '''
        return err

    def alpha(self, e):
        a = 0.5 * math.log1p((1 - e) / e)
        return a

    def normalizer(self):
        s = 0
        for weight in self.w:
            s += weight

        # return s
        return 2 * math.sqrt(self.e * (1 - self.e))

    def update_weights(self):
        for i in range(self.n):

            h = self.dif(self.h, self.x[i])
            if h != y[i]:
                print self.h, "misclassifies", i
                self.w[i] = self.w[i]/(2*self.e)
            else:
                self.w[i] = self.w[i] / (2 * (1-self.e))
            # self.w[i] = self.w[i]*(-math.e**(self.a*h*y[i]))
            # print self.w[i],
        return 0

    def dif(self, h, x):
        if h<0:
            if x[0]<=-h:
                return 1
            else:
                return -1
        else:
            if x[0]>h:
                return 1
            else:
                return -1

    def learn(self, iterations):
        for j in range(iterations):
            if self.weak_learners == []:
                return 0

            self.h, h_pos = self.pick_h()
            del self.weak_learners[h_pos]
            print self.h
            self.e = self.error(self.h)
            print "xxxxx",self.e

            # print h_pos, "has error", self.e
            if self.e > 0.5:
                break
            self.a = self.alpha(self.e)
            self.alphas.append({"h": h_pos, "alpha": self.a})
            self.update_weights()
            self.z = self.normalizer()

            # print "Z",  self.z
            for i in range(self.n):
                self.w[i] /= self.z

            print self.w

x = np.array([[0+40,10+100],[30+100,0+100],[100+100,50+100],[90+100,10+100], [0+100,70+100],[0+100,270+100], [190+100,10+100],[290+100,10+100]])

y = (1, -1, 1, 1, -1, 1, -1,-1)
n = 8.0
w = [1.0/n, 1.0/n, 1.0/n, 1.0/n, 1.0/n, 1.0/n, 1.0/n, 1.0/n]
weak_learners = [241, 141, 211, 191, 99, -241, -141, -211, -191, -99]


boost = Machine(x,y,w,weak_learners)
boost.learn(1000)
for w in boost.w:
    print w,
print "a"

for i in range(20):
    aa = 0
    p = [i*50, 0]
    for w in boost.alphas:
        aa += boost.dif(weak_learners[w['h']],p)*w['alpha']
        # print w
    print aa

