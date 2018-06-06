import math
from Tkinter import *
import bisect

"""
Python file that implements boosting machine.
"""


class Machine:
    def __init__(self, samples, desired_outputs, init_weights, weak_learners):
        """Init Machine, create Tk GUI  (800*800)"""
        self.x = samples
        self.y = desired_outputs
        self.w = init_weights
        self.weak_learners = list(weak_learners)
        self.weak_learners_backup = list(weak_learners)
        self.n = len(desired_outputs)
        self.correctly_classified = [True]*self.n
        self.z = None
        self.e = None
        self.a = None
        self.h = None
        self.alphas = []
        self.bounds = []
        self.errors = [None]*len(weak_learners)

        self.root = Tk()
        self.canvas = Canvas(self.root, width=800, height=800)

        self.canvas.pack()
        self.root.update()

        for i in range(len(self.x)):
            if self.y[i] == 1:
                self.canvas.create_oval(self.x[i][0] - 4, self.x[i][1] - 4, self.x[i][0] + 4, self.x[i][1] + 4,
                                        fill='#000000')
            else:
                self.canvas.create_oval(self.x[i][0] - 4, self.x[i][1] -4, self.x[i][0] + 4, self.x[i][1] + 4)
            self.canvas.create_text(self.x[i][0], self.x[i][1] + 13, text=str(self.x[i][0]), tags="text")
        self.canvas.pack()
        self.root.update()

    def draw_boundaries(self):
        """Function only to draw selected weak learners (classifier)"""
        for i in range(len(self.bounds)):
            self.canvas.create_line(abs(self.bounds[i][0]), 0, abs(self.bounds[i][0]), 750)
            self.canvas.create_text(abs(self.bounds[i][0]), 780, text=str(self.bounds[i][0]), tags="text")

        self.canvas.pack()
        self.root.update()

    def pick_h(self):
        """Function that returns the best weak learner (classifier) from remaining. based on weighted error"""
        min_err = 1000
        min_err_pos = None
        for i in range(len(self.weak_learners)):
            if self.weak_learners[i] is not None:
                self.errors[i] = self.error(self.weak_learners[i])
                if self.errors[i] < min_err:
                    min_err = self.errors[i]
                    min_err_pos = i

        print self.weak_learners[min_err_pos], "min error ",min_err
        return self.weak_learners[min_err_pos], min_err_pos

    def error(self, h):
        """Function that calculates error of the classifier h"""
        err = 0
        for i in range(len(self.x)):
            if self.dif(h, self.x[i]) != self.y[i]:
                err += self.w[i]

        return err

    @staticmethod
    def alpha(e):
        """Returns alpha, the minimizer of the exponential error function for Discrete AdaBoost (wiki)"""
        # print "alpha", 0.5 * math.log1p((1 - e) / e)
        return 0.5 * math.log1p((1 - e) / e)

    def normalizer(self):
        """Function that returns sum of all weights, so they can be normalized"""
        s = 0
        for weight in self.w:
            s += weight
        return s

    def update_weights(self):

        for i in range(self.n):
            h = self.dif(self.h, self.x[i])
            if h != self.y[i]:
                # print self.h, "missclassifies", i
                self.w[i] = self.w[i] / (2 * self.e)
            else:
                self.w[i] = self.w[i] / (2 * (1 - self.e))

        return 0

    @staticmethod
    def dif(h, x):
        """Static method that returns the sign, based on which side of the classifier h is the x[0] value,
            when h<0, that means the signs on the sides of classifier h are flipped"""
        if h < 0:
            if x[0] <= -h:
                return 1
            else:
                return -1
        else:
            if x[0] > h:
                return 1
            else:
                return -1

    def learn(self, iterations):
        """Main boosting function, selects classifiers and adds them to self.bounds array"""
        self.normalize_classifiers()
        for j in range(iterations):
            self.h, h_pos = self.pick_h()
            self.e = self.error(self.h)
            # needs to be updated
            '''
            if self.weak_learners[h_pos] is not None:
                if h_pos < len(self.weak_learners)-1 and self.weak_learners[h_pos+1] is not None:
                    if self.weak_learners[h_pos] == -self.weak_learners[h_pos+1]:
                        self.weak_learners[h_pos + 1] = None
                if h_pos > 0 and self.weak_learners[h_pos-1] is not None:
                    if self.weak_learners[h_pos] == -self.weak_learners[h_pos-1]:
                        self.weak_learners[h_pos - 1] = None
            '''
            self.weak_learners[h_pos] = None
            print self.e >= 0.5
            if (self.e >= 0.5) or (self.e is None):
                print "not used classifiers:", sorted(self.weak_learners)
                break

            # print self.h, "picked with error ", self.e
            self.bounds.append([self.h, h_pos])

            self.a = self.alpha(self.e)
            self.alphas.append({"h": h_pos, "alpha": self.a})

            self.z = self.normalizer()
            self.update_weights()

            # renormalize weights
            for i in range(self.n):
                self.w[i] /= self.z
            # print self.w, sum(self.w)

        # draw all selected classifiers
        self.draw_boundaries()

    def normalize_classifiers(self):
        """Function to delete all unrelated classifiers"""

        dic = {}
        for i in range(self.n):
            if self.x[i][0] in dic:
                if self.y[i] not in dic[self.x[i][0]]:
                    dic[self.x[i][0]][self.y[i]] = 1
            else:
                dic[self.x[i][0]] = {self.y[i]: 1}
        helper = sorted(zip(*self.x)[0], key = lambda a: a)

        for i in range(0,len(self.weak_learners),2):
            pos1 = bisect.bisect_left(helper, self.weak_learners[i])-1
            pos2 = bisect.bisect_right(helper, self.weak_learners[i])
            if pos1 >= 0 and pos2<self.n:
                if (1 in dic[helper[pos1]]) ^ (-1 in dic[helper[pos1]]) \
                        and ((1 in dic[helper[pos2]]) ^ (-1 in dic[helper[pos2]])):
                    for key in dic[helper[pos1]]:
                        for key1 in dic[helper[pos2]]:
                            if key == key1:
                                print "deleting", self.weak_learners[i],key,key1
                                self.weak_learners[i] = None
                                self.weak_learners[i+1] = None

    def classify(self, point):
        result = 0
        for w in self.alphas:
            result += self.dif(self.weak_learners_backup[w['h']], point) * w['alpha']
        # print boost.dif(self.weak_learners_backup[w['h']],p)*w['alpha'],
        # print w
        print result
        return result

    def classify_gui_point(self,event):
        x = float(event.x)
        y = float(event.y)
        point = [x,y]
        if self.classify(point) > 0:
            self.canvas.create_oval(x, y, x + 5, y + 5, fill='#000000')
            self.canvas.create_text(x, y, text="+", tags="text")
        else:
            self.canvas.create_oval(x, y, x + 5, y + 5)
            self.canvas.create_text(x, y, text="-", tags="text")
