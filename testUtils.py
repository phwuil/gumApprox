# -*- coding: utf-8 -*-
import pyAgrum as gum

from ApproxInference.utils import KL, compactPot, draw
from ApproxInference.probabilityEstimator import ProbabilityEstimator

def compareApprox(m, bn, evs):
  """
  Compare results in m with a LazyPropagation on bn with evidence evs

  :param m: the approximated algorithm
  :param bn: the bayesian network
  :param evs: the dict of evidence
  :return: void
  """
  ie = gum.LazyPropagation(bn)
  ie.setEvidence(evs)
  ie.makeInference()

  res = []
  for i in bn.ids():
    v, c = m.results(i)
    if v is not None:
      res.append((bn.variable(i).name(),
                  KL(ie.posterior(i), v),
                  compactPot(ie.posterior(i)),
                  compactPot(v),
                  c))
    else:
      print("{}: {}".format(bn.variable(i).name(), evs[bn.variable(i).name()]))

  for r in sorted(res, key=lambda item: item[1], reverse=True):
    print("{} : {:3.5f}\n        exact  : {}\n        approx : {}  ({:7.5f})".format(*r))


def testRandom():
  v = gum.LabelizedVariable("a", "a", 6)
  p = gum.Potential().add(v).fillWith([1, 2, 4, 5, 3, 2]).normalize()
  particle = ProbabilityEstimator(v)
  print(p)
  for i in range(10000):
    w, q = draw(p)
    particle.add(q)
  print("with {} draws, results are".format(particle._nbr))
  print(particle.value())

if __name__ == '__main__':
  testRandom()
