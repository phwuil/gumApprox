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
  Compute KL(p,q), Kullback-Leibler divergence

  :param p: gum.Potential
  :param q: gum.Potential
  :return: float
  """
  Ip = gum.Instantiation(p)
  Iq = gum.Instantiation(q)
  s = 0
  while not Ip.end():
    if p.get(Ip) > 0:
      s += p.get(Ip) * math.log2(p.get(Ip) / q.get(Iq))
    Ip.inc()
    Iq.inc()

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
    res += "|{:7.5f}".format(100 * p.get(i))
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


def conditionalModel(bn, evs):
  """
  create a new condtional bn from a bn and a instanticiation of some variable

  :param bn: a bayesian network
  :param evs: map of evidence
  :return: a bayesian network
  """
  newbn = gum.BayesNet(bn)
  newevs = dict(evs)
  for name in evs:
    nid = newbn.idFromName(name)
    for ch in newbn.children(nid):
      # create the new cpt
      q = newbn.cpt(ch) \
        .extract({name: evs[name]}) \
        .reorganize([v.name() for v in bn.cpt(ch).variablesSequence() if v.name() != name])
      # erase arc
      newbn.eraseArc(nid, ch)
      # update cpt
      newbn.cpt(ch)[:] = q[:]

    # remove evidence without parent
    if len(newbn.parents(nid)) == 0:
      newbn.erase(nid)
      newevs.pop(name)

  return newbn, newevs


def mutilatedModel(bn, evs):
  """
  mutilate a bayesian network : remove all variable with evidence from the BN
  :param bn: a bayesian network
  :param evs: map of evidence
  :return: a mutilated bayesian network
  """
  newbn, newevs = conditionalModel(bn, evs)  # now, all evidence has no children

  for name in newevs:
    newbn.erase(newbn.idFromName(name))

  return newbn  # now, all evidence is removed from the bayesian network


def unsharpenedModel(bn, value=0.1):
  """
  Modify the cpts of a BN in order to tend to uniform distributions

  :param bn: a bayesian netwok
  :param value: a value that will be added every where before normalization
  :return: the newBN
  """
  newbn = gum.BayesNet(bn)
  for i in newbn.ids():
    newbn.cpt(i).translate(value).normalize()

  return newbn


def minParamInModel(bn):
  """
  return the smallest param in a bayesian network

  :param bn: a bayesian network
  :return: the smallest param
  """
  return min([bn.cpt(i).min() for i in bn.ids()])
