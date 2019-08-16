from itertools import islice
import simplejson
import sys, os, subprocess
import PIL
import io
import numpy as np
from pylab import *
import sys
import torch
import cctc2 as cctc

def test_square():
    a = torch.randn(3, 3)
    print(a)
    cctc.square(a)
    print(a)

def test_resize():
    a = torch.randn(3, 3)
    cctc.make_one(a)
    print(a)

def rownorm(t):
    denom = t.sum(1).unsqueeze(1).repeat(1, t.size(1))
    result = t / denom
    return result

def batch_rownorm(t):
    for i in range(len(t)):
        t.select(0, i).copy_(rownorm(t.select(0, i)))
    return t

def test_align():
    a = rownorm(torch.rand(100, 17))
    b = rownorm(torch.rand(20, 17))
    c = cctc.ctc_align_targets(a, b)
    print(c.size())

def test_align_batches():
    a = batch_rownorm(torch.rand(3, 100, 17))
    b = batch_rownorm(torch.rand(3, 20, 17))
    c = cctc.ctc_align_targets_batch(a, b)
    print(c.size())
