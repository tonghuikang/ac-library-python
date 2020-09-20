import sys
import heapq, functools, collections
import math, random
from collections import Counter, defaultdict

# from atcoder.fenwicktree import FenwickTree

# available on Google, not available on Codeforces
# import numpy as np
# import scipy

# from atcoder.math import crt


def prime_factors(nr):
    i = 2
    factors = []
    while i <= nr:
        if i > math.sqrt(nr):
            i = nr
        if (nr % i) == 0:
            factors.append(int(i))
            nr = nr / i
        elif i == 2:
            i = 3
        else:
            i = i + 2
    return factors


def solve(k):  # fix inputs here
    console("----- solving ------")

    # console(prime_factors(k))
    # console(prime_factors(1100144))
    # console(prime_factors(1100144+1))
    # console((1100144+1)*(1100144)%7575345)

    # y = crt([1,0], [8,k//8])
    # console(k,y)

    # let z be a multple of 2k
    # check if 8z+1 is a square
    for z in range(8*k+1, 10**25, 8*k):
        y = z
        if math.isqrt(y)**2 == y:
            res = (math.isqrt(y) - 1) // 2
            # console(prime_factors(y))
            # console(prime_factors(res))
            # console(prime_factors(res+1))
            # console(prime_factors((res+1)*res))
            return res

    return -1




def console(*args):  # the judge will not read these print statement
    print('\033[36m', *args, '\033[0m', file=sys.stderr)
    return

# fast read all
# inp = sys.stdin.readlines()

for case_num in [1]:
    # read line as a string
    # strr = input()

    # read line as an integer
    k = int(input())
    
    # read one line and parse each word as a string
    # lst = input().split()

    # read one line and parse each word as an integer
    # lst = list(map(int,input().split()))

    # read matrix and parse as integers (after reading read nrows)
    # lst = list(map(int,input().split()))
    # nrows = lst[0]  # index containing information, please change
    # grid = []
    # for n in range(nrows):
    #     grid.append(list(map(int,inp[n+1].split())))

    res = solve(k)  # please change
    
    # res = "\n".join(str(x) for x in res)
    # Google - case number required
    # print("Case #{}: {}".format(case_num+1, res))

    # Codeforces - no case number required
    print(res)
