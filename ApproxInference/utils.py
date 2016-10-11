# -*- coding: utf-8 -*-
import math
import random

import pyAgrum as gum


def deterministicPotential(v, val):
  l = [0] * v.domainSize()
  l[val] = 1
  return gum.Potential().add(v).fillWith(l)


def uniformPotential(v):
  return gum.Potential().add(v).fillWith(1).normalize()


def KL(p, q):
  """
  Compute KL(p,k), Kullback-Leibler divergence

  :param p: gum.Potential
  :param q: gum.Potential
  :return: float
  """
  I = gum.Instantiation(p)
  s = 0
  while not I.end():
    if p.get(I) > 0:
      s += p.get(I) * math.log2(p.get(I) / q.get(I))
    I.inc()

  # it may happen that if p==q, s<0 (approximation)
  return 0 if s <= 0 else s


def draw(p):
  """
  Draw a sample using p
  :param p: a probability distribution over a variable v
  :return: (v,q) where v is the value and q is the deterministic distribution for v
  """
  r = random.random()
  i = gum.Instantiation(p)
  val = 0
  while not i.end():
    r -= p.get(i)
    i.inc()
    if r <= 0:
      val = i.val(0)
      break

  return val, deterministicPotential(p.variable(0), val)


def argmax(iterable):
  """
  return the argmax on an iterable
  :param iterable:
  :return: the argmax
  """
  return max(enumerate(iterable), key=lambda x: x[1])[0]


def compactPot(p):
  res = ""
  i = gum.Instantiation(p)
  i.setFirst()
  while not i.end():
    res += "|{:7.3f}".format(100 * p.get(i))
    i.inc()
  return "[" + res[1:] + "]"


def isAlmostEqualPot(p1, p2):
  """
  check if the 2 potentials have the same parameters (even if not on the same variables)

  :param p1:
  :param p2:
  :return: boolean
  """
  q = p1[:] - p2[:]
  r = math.sqrt((q * q).max())
  return r < 1e-5


def mutilate(bn, evs):
  """
  create a new bn from a bn and a instantication of some variable

  :param bn:
  :param evs:
  :return:
  """
  for name in evs:
    nid = bn.idFromName(name)
    for ch in bn.children(nid):
      # create the new cpt
      q = bn.cpt(ch) \
        .extract({name: evs[name]}) \
        .reorganize([v.name() for v in bn.cpt(ch).variablesSequence() if v.name() != name])
      # erase arc
      bn.eraseArc(nid, ch)
      # update cpt
      bn.cpt(ch)[:] = q[:]
  return bn
