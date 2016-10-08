# -*- coding: utf-8 -*-
from . import utils
from .probabilityEstimator import ProbabilityEstimator


class GenericInference:
  """
  Base class for all inference with hard evidence
  """

  def __init__(self, bn, evs, verbose=True):
    self._originalbn = bn
    self._evs = evs
    self._bn = utils.mutilate(bn, evs)
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
    return None if i not in self._estimators else self._estimators[i].value()

  def results(self, i):
    if i not in self._estimators:
      return None, None
    else:
      return self._estimators[i].value(), self._estimators[i].confidence()
