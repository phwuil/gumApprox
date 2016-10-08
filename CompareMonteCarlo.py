# -*- coding: utf-8 -*-
import pyAgrum as gum

import ApproxInference.utils
from ApproxInference.probabilityEstimator import ProbabilityEstimator


class MonteCarlo(object):
  def __init__(self, bn):
    self._bn = bn
    self._estimators = {i: ProbabilityEstimator(self._bn.variable(i)) for i in self._bn.ids()}
    self._hardEstimators = {i: ProbabilityEstimator(self._bn.variable(i)) for i in self._bn.ids()}

  @staticmethod
  def multipleRound(bn, size=1000):
    estimators = {i: ProbabilityEstimator(bn.variable(i)) for i in bn.ids()}
    hardEstimators = {i: ProbabilityEstimator(bn.variable(i)) for i in bn.ids()}

    def oneRound():
      current = {}
      for i in bn.topologicalOrder():
        q = gum.Potential(bn.cpt(i))
        for j in bn.parents(i):
          q *= current[j]
        q = q.margSumIn([bn.variable(i).name()])
        v,r = ApproxInference.utils.draw(q)
        current[i] = r
        estimators[i].add(q)
        hardEstimators[i].add(r)

    for i in range(size):
      oneRound()
    return estimators, hardEstimators

  def run(self, arret=1e-2, size=1000, verbose=False):
    while True:
      est, hest = self.multipleRound(self._bn, size)
      for node_id, estimator in est.items():
        self._estimators[node_id] += estimator
        self._hardEstimators[node_id] += hest[node_id]

      x = max([self._estimators[i].confidence() for i in self._bn.ids()])
      hx = max([self._hardEstimators[i].confidence() for i in self._bn.ids()])
      if verbose:
        print("confidence : {} {} => {}%".format(x, hx, 100 * (hx - x) / x))

      if x < arret:
        break

  def posterior(self, i):
    return self._estimators[i].value()

  def everything(self, i):
    return self._estimators[i].value(), self._estimators[i].confidence(), self._hardEstimators[i].value(), \
           self._hardEstimators[i].confidence()


def main():
  bn = gum.loadBN("alarm.bif")

  ie = gum.LazyPropagation(bn)
  ie.makeInference()

  m = MonteCarlo(bn)
  m.run(1e-2, 300, verbose=True)

  print("done")

  for i in bn.ids():
    ev, ec, hv, hc = m.everything(i)
    print("{}: {} ({}) =!= {} ({})".format(bn.variable(i).name(), ev, ec, hv, hc))


if __name__ == '__main__':
  main()
