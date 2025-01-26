import numpy as ql

# R is The Reward Matrix for each state
R = ql.matrix([ [0,0,0,0,1,0],
                [0,0,0,1,0,1],
                [0,0,100,1,0,0],
                [0,1,1,0,1,0],
                [1,0,0,1,0,0],
                [0,1,0,0,0,0] ])

Q = ql.matrix(ql.zeros([6,6]))

adexe = ql.array([[0, 0, 0, 1, 0, 1]])
# ql.where(adexe > 0, adexe)
# result = [ xv if c else yv for c, xv, yv in (xv > 0 < 5, adexe)]

# x, y = ql.ogrid[:3 , :4]
# print(x,y)

print(type(adexe))
# print(type(R[1]))
print(ql.asarray(adexe > 0))
