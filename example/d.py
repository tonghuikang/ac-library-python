import sys
import heapq, functools, collections
import math, random
from collections import Counter, defaultdict

# from atcoder.fenwicktree import FenwickTree

# available on Google, not available on Codeforces
# import numpy as np
# import scipy


def solve(arr, k):  # fix inputs here
    console("----- solving ------")
    console(arr,k)

    

    return 0




def console(*args):  # the judge will not read these print statement
    print('\033[36m', *args, '\033[0m', file=sys.stderr)
    return

# fast read all
inp = sys.stdin.readlines()

for case_num in [1]:
    # read line as a string
    # strr = input()

    # read line as an integer
    # nrows = int(inp[0])
    
    # read one line and parse each word as a string
    # lst = input().split()

    # read one line and parse each word as an integer
    nrows, k = list(map(int,inp[0].split()))

    # read matrix and parse as integers (after reading read nrows)
    # lst = list(map(int,input().split()))
    # nrows = lst[0]  # index containing information, please change
    grid = []
    for n in range(nrows):
        grid.append(list(map(int,inp[n+1].split()))[0])

    res = solve(grid, k)  # please change
    
    # res = "\n".join(str(x) for x in res)
    # Google - case number required
    # print("Case #{}: {}".format(case_num+1, res))

    # Codeforces - no case number required
    print(res)
