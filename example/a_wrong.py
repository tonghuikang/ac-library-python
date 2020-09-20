import sys
import heapq, functools, collections
import math, random
from collections import Counter, defaultdict

from atcoder.fenwicktree import FenwickTree

# available on Google, not available on Codeforces
# import numpy as np
# import scipy


def solve(grid, n):  # fix inputs here
    console("----- solving ------")
    grid = [(x-1, y-1) for x,y in grid]
    m = {}
    for i,(x,y) in enumerate(grid):
        m[x] = i
    grid = sorted(grid)
    console(grid)

    fenwick_tree = FenwickTree(n+1)

    res = [0 for _ in range(n)]

    for i,(x,y) in enumerate(grid):
        fenwick_tree.add(y, 1)
        add = fenwick_tree.sum(0, y)
        print(i,x,y,add)
        res[x] += add

    print()
    fenwick_tree = FenwickTree(n+1)

    for i,(x,y) in list(enumerate(grid))[::-1]:
        add = fenwick_tree.sum(y, n)
        fenwick_tree.add(y, 1)
        print(i,x,y,add)
        res[x] += add

    console(res)
    
    res2 = [0 for _ in range(n)]
    for i,x in m.items():
        res2[x] = res[i]
    return res2


def console(*args):  # the judge will not read these print statement
    print('\033[36m', *args, '\033[0m', file=sys.stderr)
    return

# fast read all
inp = sys.stdin.readlines()

for case_num in [1]:
    # read line as a string
    # strr = input()

    # read line as an integer
    nrows = int(inp[0])
    
    # read one line and parse each word as a string
    # lst = input().split()

    # read one line and parse each word as an integer
    # lst = list(map(int,input().split()))

    # read matrix and parse as integers (after reading read nrows)
    # lst = list(map(int,input().split()))
    # nrows = lst[0]  # index containing information, please change
    grid = []
    for n in range(nrows):
        grid.append(list(map(int,inp[n+1].split())))

    res = solve(grid, nrows)  # please change
    
    res = "\n".join(str(x) for x in res)
    # Google - case number required
    # print("Case #{}: {}".format(case_num+1, res))

    # Codeforces - no case number required
    print(res)
