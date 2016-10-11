# -*- coding: utf-8 -*-
import random

import pyAgrum as gum

from . import utils
from .infGenericSampler import GenericSamplerInference
from .probabilityEstimator import ProbabilityEstimator


class Gibbs(GenericSamplerInference):
  def __init__(self, bn, evs, verbose=True):
    super().__init__(bn, evs, verbose)

  @staticmethod
  def burnin(bn, evs=None, size=100):
    currentPotentials = {}
    for i in bn.ids():
      if bn.variable(i).name() not in evs:
        v, currentPotentials[i] = utils.draw(utils.uniformPotential(bn.variable(i)))
      else:
        currentPotentials[i] = utils.deterministicPotential(bn.variable(i), evs[bn.variable(i).name()])

    def oneRoundBurnin(bn):
      ids = bn.ids()[:]
      random.shuffle(ids)
      for i in ids:
        if bn.variable(i).name() not in evs:
          q = gum.Potential(bn.cpt(i))
          for u in bn.parents(i):
            q *= currentPotentials[u]

          for j in bn.children(i):
            qj = gum.Potential(bn.cpt(j))
            qj *= currentPotentials[j]
            for k in bn.parents(j):
              if k != i:
                qj *= currentPotentials[k]
            q *= qj

          q = q.margSumIn([bn.variable(i).name()])
          q.normalizeAsCPT()

          v, currentPotentials[i] = utils.draw(q)

    for i in range(size):
      oneRoundBurnin(bn)

    return currentPotentials

  def multipleRound(self, initial_current, size=1000):
    estimators = {i: ProbabilityEstimator(self._bn.variable(i)) for i in self._bn.ids() if
                  self._bn.variable(i).name() not in self._evs}
    currentPotentials = {k: gum.Potential(initial_current[k]) for k in initial_current.keys()}

    def oneRound():
      ids = self._bn.ids()[:]
      random.shuffle(ids)
      # todo: select only some ids and not all
      for i in ids:
        name = self._bn.variable(i).name()
        if name not in self._evs:
          q = gum.Potential(self._bn.cpt(i))
          for u in self._bn.parents(i):
            q *= currentPotentials[u]

          for j in self._bn.children(i):
            qj = gum.Potential(self._bn.cpt(j))
            qj *= currentPotentials[j]
            for k in self._bn.parents(j):
              if k != i:
                qj *= currentPotentials[k]
            q *= qj

          q = q.margSumIn([name])
          q.normalizeAsCPT()

          v, currentPotentials[i] = utils.draw(q)
          estimators[i].add(q)

    for i in range(size):
      oneRound()

    return currentPotentials, estimators

  def run(self, epsilon=1e-2, size=1000):
    """
    :rtype: void
    """
    if self._verbose:
      print("Burn in")
    currentPotentials = self.burnin(self._bn, self._evs)

    print("Looping")
    while True:
      currentPotentials, estimators = self.multipleRound(currentPotentials, size)
      for node_id, estimator in estimators.items():
        self._estimators[node_id] += estimator

      x = max([self._estimators[i].confidence() for i in self._estimators])
      argx, x = max([(i, self._estimators[i].confidence()) for i in self._estimators.keys()], key=lambda x: x[1])
      if self._verbose:
        print("confidence : {:12.9f} ({})".format(x, self._bn.variable(argx).name()))

      if x < epsilon:
        break
