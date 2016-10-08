# -*- coding: utf-8 -*-
from multiprocessing import Pool

import pyAgrum as gum

import utils
from probabilityEstimator import ProbabilityEstimator


def multipleRound(bn, size=1000):
  estimators = {i: ProbabilityEstimator(bn.variable(i)) for i in bn.ids()}

  def oneRound():
    current = {}
    for i in bn.topologicalOrder():
      q = gum.Potential(bn.cpt(i))
      for j in bn.parents(i):
        q *= current[j]
      q = q.margSumIn([bn.variable(i).name()])
      v,current[i] = utils.draw(q)
      estimators[i].add(q)

  for i in range(size):
    oneRound()
  return estimators

class ParallelMonteCarlo(object):
  def __init__(self, bn):
    self._bn = bn
    self._estimators = {i: ProbabilityEstimator(self._bn.variable(i)) for i in self._bn.ids()}




  def run(self, arret=1e-2, size=1000,verbose=False):
    nbproc=4
    while True:
      pool = Pool(processes=nbproc)
      res=[(self._bn,size)]*nbproc
      for e in pool.map(multipleRound,res):
        for node_id,estimator in e.items():
          self._estimators[node_id]+=estimator

      x = max([self._estimators[i].confidence() for i in self._bn.ids()])
      if verbose:
        print("confidence : {}".format(x))

      if x < arret:
        break

  def posterior(self, i):
    return self._estimators[i].value()


def main():
  bn = gum.loadBN("alarm.bif")

  ie = gum.LazyPropagation(bn)
  ie.makeInference()

  m = ParallelMonteCarlo(bn)
  m.run(1e-2, 300,verbose=True)

  print("done")
  for i in bn.ids():
    print("{}: {}".format(bn.variable(i).name(), int(100000 * utils.KL(ie.posterior(i), m.posterior(i))) / 1000))


if __name__ == "__main__":
  main()
