import numpy as np
import matplotlib.pyplot as plt
import math
from src import ajustar_min_cuadrados


def gq2(xs, ys):
    c2 = sum(x**4 for x in xs)
    c1 = sum(x**3 for x in xs)
    c0 = sum(x**2 for x in xs)
    cind = sum((x**2)*y for x, y in zip(xs, ys))
    return (c2, c1, c0, cind)


def gq1(xs, ys):
    c2 = sum(x**3 for x in xs)
    c1 = sum(x**2 for x in xs)
    c0 = sum(x for x in xs)
    cind = sum(x*y for x, y in zip(xs, ys))
    return (c2, c1, c0, cind)


def gq0(xs, ys):
    c2 = sum(x**2 for x in xs)
    c1 = sum(x for x in xs)
    c0 = len(xs)
    cind = sum(ys)
    return (c2, c1, c0, cind)


xs1 = [-5.0,-3.8889,-2.7778,-1.6667,-0.5556,0.5556,1.6667,2.7778,3.8889,5.0]
ys1 = [57.2441,33.0303,16.4817,7.0299,0.5498,0.7117,3.4185,12.1767,24.9167,44.2495]

a2, a1, a0 = ajustar_min_cuadrados(xs1, ys1, gradiente=[gq2, gq1, gq0])

x = np.linspace(-5, 5, 200)
y = [a2*xi**2 + a1*xi + a0 for xi in x]

plt.scatter(xs1, ys1)
plt.plot(x, y)
plt.show()

print(a2, a1, a0)
print(a2*2.25**2 + a1*2.25 + a0)
print(a2*(-2.25)**2 + a1*(-2.25) + a0)


def ge1(xs, ys):
    c1 = 0
    c0 = 0
    cind = 0
    for x, y in zip(xs, ys):
        ex = math.exp(x)
        c1 += ex*ex
        c0 += ex
        cind += y*ex
    return (c1, c0, cind)


def ge0(xs, ys):
    c1 = sum(math.exp(x) for x in xs)
    c0 = len(xs)
    cind = sum(ys)
    return (c1, c0, cind)


xs2 = [0.0003,0.0822,0.2770,0.4212,0.4403,0.5588,0.5943,0.6134,0.9070,1.0367,1.1903,1.2511,1.2519,1.2576,1.6165,1.6761,2.0114,2.0557,2.1610,2.6344]
ys2 = [1.1017,1.5021,0.3844,1.3251,1.7206,1.9453,0.3894,0.3328,1.2887,3.1239,2.1778,3.1078,4.1856,3.3640,6.0330,5.8088,10.5890,11.5865,11.8221,26.5077]

a, b = ajustar_min_cuadrados(xs2, ys2, gradiente=[ge1, ge0])

x2 = np.linspace(min(xs2), max(xs2), 200)
y2 = [a*math.exp(b*xi) for xi in x2]

plt.scatter(xs2, ys2)
plt.plot(x2, y2)
plt.show()

print(a, b)
print(a*math.exp(b*1))
print(a*math.exp(b*5))
