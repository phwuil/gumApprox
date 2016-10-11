# -*- coding: utf-8 -*-
import pyAgrum as gum

from . import utils
from .infGenericSampler import GenericSamplerInference
from .probabilityEstimator import ProbabilityEstimator


class Weighted(GenericSamplerInference):
  def __init__(self, bn, evs, verbose=True):
    super().__init__(bn, evs, verbose)
    self._nbrReject = 0
    self._nbr = 0

  def multipleRound(self, size=1000):
    estimators = {i: ProbabilityEstimator(self._bn.variable(i))
                  for i in self._bn.ids()
                  if self._bn.variable(i).name() not in self._evs}

    def oneRound():
      self._nbr += 1
      currentPotentials = {}
      proba = {}

      # simple sampling w.r.t. the BN
      inst = {}
      for i in self._bn.topologicalOrder():
        name = self._bn.variable(i).name()
        q = gum.Potential(self._bn.cpt(i))
        for j in self._bn.parents(i):
          q *= currentPotentials[j]
        q = q.margSumIn([name])
        inst[name], currentPotentials[i] = utils.draw(q)
        if name not in self._evs:
          proba[i] = q

      # forcing the value of evidence and compute the probability of this forces instance
      globalProba = 1.0
      for i in self._bn.topologicalOrder():
        name = self._bn.variable(i).name()
        if name in self._evs:
          inst[name] = self._evs[name]
        localp = self._bn.cpt(i)[inst]
        if localp == 0:
          return False
        globalProba *= localp

      print("{} : {}",inst,globalProba)
      for i in proba.keys():
        estimators[i].add(proba[i], globalProba)
      return True

    for i in range(size):
      while not oneRound():
        pass
    return estimators

  def run(self, epsilon=1e-2, size=1000):
    """
    :rtype: void
    """
    self._nbrReject = 0
    self._nbr = 0
    print("Looping")
    while True:
      for node_id, estimator in self.multipleRound(size).items():
        self._estimators[node_id] += estimator

        argX, x = max([(i, self._estimators[i].confidence()) for i in self._estimators.keys()], key=lambda x: x[1])

      if self._verbose:
        print("confidence : {:12.9f} ({})".format(x, self._bn.variable(argX).name()))
        print("    Reject : {:8.4f}%".format(100 * self._nbrReject / self._nbr))

      if x < epsilon:
        break
