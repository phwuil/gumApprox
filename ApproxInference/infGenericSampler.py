# -*- coding: utf-8 -*-
from . import utils
from .probabilityEstimator import ProbabilityEstimator


class GenericInference:
  """
  Base class for all inference with hard evidence
  """

  def __init__(self, bn, evs, verbose=True):
    self._originalbn = bn
    self._originalevs = evs
    self._bn, self._evs = utils.mutilate(bn, evs)
    self._verbose = verbose


class GenericSamplerInference(GenericInference):
  """
  Base class for all inference with hard evidence
  """

  def __init__(self, bn, evs, verbose=True):
    super().__init__(bn, evs, verbose)
    self._estimators = {i: ProbabilityEstimator(self._bn.variable(i)) for i in self._bn.ids() if
                        bn.variable(i).name() not in evs}

  def posterior(self, i):
    v = self._originalbn.variable(i)
    n = v.name()
    if n in self._originalevs:
      return utils.deterministicPotential(v, self._originalevs[n])
    else:
      return self._estimators[i].value()

  def results(self, i):
    if self._originalbn.variable(i).name() in self._originalevs:
      return utils.deterministicPotential(self._originalbn.variable(i),
                                          self._originalevs[self._originalbn.variable(i).name()]) \
        , 1.0
    else:
      return self._estimators[i].value() \
        , self._estimators[i].confidence()
