# -*- coding: utf-8 -*-
import math

import pyAgrum as gum


class ProbabilityEstimator():
  def __init__(self, v):
    """
    create a particule for the variable V
    """
    self._val = gum.Potential().add(v).fillWith(0)
    self._val2 = gum.Potential().add(v).fillWith(0)
    self._nbr = 0
    self._name = [v.name()]

  def add(self, p, w=None):
    """
    add a new Potential p
    """
    if p.var_names != self._name:
      raise NameError("Trying to add a potential on {} instead of {}".format(p.var_names, self._name))

    if w is None:
      q = p
      w = 1
    else:
      q = p.scale(w)

    self._val += q
    self._val2 += q * q
    self._nbr += w

  def __add__(self, other):
    if self._name != other._name:
      raise NameError("Trying to add a ProbabilityEstimator on {} instead of {}".format(other._name, self._name))
    self._val += other._val
    self._val2 += other._val2
    self._nbr += other._nbr
    return self

  def value(self):
    q = gum.Potential(self._val)
    if self._nbr != 0:
      q.scale(1.0 / self._nbr)
    return q

  def confidence(self, pval=1.96):
    if self._nbr == 0:
      return 100

    v = gum.Potential(self._val)
    v.scale(1.0 / self._nbr)

    v2 = gum.Potential(self._val2)
    v2.scale(1.0 / self._nbr)

    return pval * math.sqrt((v2 - v * v).max() / self._nbr)


def main(arret=1e-3):
  domainSize = 5
  v = gum.LabelizedVariable("a", "a", domainSize)
  s = ProbabilityEstimator(v)

  p = gum.Potential().add(v)
  while True:
    for k in range(1000):
      p.fillWith(gum.randomDistribution(domainSize))
      s.add(p)
    print(s._nbr)
    if s.confidence() < arret:
      break

  print(s.value())


if __name__ == "__main__":
  main()
