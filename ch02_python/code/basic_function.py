#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 18:05:12 2018

@author: tgbaggio
"""


# 定义函数f
def f(a, b):
    return a + b

f(1, 2)

# lambda表达式定义与f一摸一样的函数
g = lambda a, b: a + b

g(1, 2)


l = [1, 2, 3]


def h(a):
    return a + 1

# 使用map和函数h对l里面的每一个元素加1
list(map(h, l))

# map加上lambda表达式可以达到同样的效果
list(map(lambda a: a + 1, l))

# 使用filter对数据进行过滤
list(filter(lambda a: a >= 2, l))

# 使用reduce对数据进行加和
from functools import reduce
reduce(lambda accvalue, newvalue: accvalue + newvalue, l, 10)
