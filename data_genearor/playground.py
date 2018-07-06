# a=[("13.5",100),("16.5", 100)]
# b=[("14.5",100), ("15.5", 100)]
# c=[("15.5",100), ("16.5", 100)]
# input=[a,b,c]
#
# from collections import Counter
#
# counters = [Counter(dict(x)) for x in input]
# print counters
# print sum(counters,Counter())
# # print sum(
# #     counters,
# #     Counter())

# import numpy as np
# import torch
# x = torch.Tensor([[1,2],[3,4]])
# print x.shape
# y =torch.Tensor([[5,6],[7,8],[9,10]])
# print y.shape
# n = x.size(0)
# m = y.size(0)
# d = x.size(1)
#
# x = x.unsqueeze(1).expand(n, m, d)
# print x
# y = y.unsqueeze(0).expand(n, m, d)
# print y
#
# dist = torch.pow(x - y, 2).sum(2)
# print dist
# values, indices = torch.max(dist, 1)
# print values
# print indices
#
# print "done"

import numpy as np
import torch
x = torch.randn(3, 4)
print x

indices = torch.tensor([0, 2])
a= torch.index_select(x, 0, indices)
print a
print a[1:]

print torch.index_select(x, 1, indices)[2:]
