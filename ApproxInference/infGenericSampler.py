# -*- coding: utf-8 -*-
import random

from . import utils
from .probabilityEstimator import ProbabilityEstimator


class GenericInference:
  """
  Base class for all inference with hard evidence
  """

  def __init__(self, bn, evs, verbose=True):
    random.seed()
    if verbose:
      print("Initiazing random sampler : {}".format(random.random()))

    self._originalBN = bn
    self._originalEvs = evs
    self._bn, self._evs = utils.conditionalModel(bn, evs)
    self._verbose = verbose
    if self._verbose:
      print("Conditioning done")
      print("  - dimension from {} to {}".format(self._originalBN.dim(), self._bn.dim()))
      print("  - evs from {} to {}".format(len(self._originalEvs), len(self._evs)))

  def posterior(self, i):
    raise NotImplementedError("posteriori(self,i) has to be implemented by your inference")

  def results(self, i):
    return self.posterior(i), -1  # complete unconfidence


class GenericSamplerInference(GenericInference):
  """
  Base class for all inference with hard evidence
  """

  def __init__(self, bn, evs, verbose=True):
    super().__init__(bn, evs, verbose)
    self._estimators = {i: ProbabilityEstimator(self._bn.variable(i)) for i in self._bn.ids() if
                        bn.variable(i).name() not in evs}

  def posterior(self, i):
    """
    return a posterior for node i

    :param i: the nodeId
    :return:  a gum.Potential
    """
    v = self._originalBN.variable(i)
    n = v.name()
    if n in self._originalEvs:
      return utils.deterministicPotential(v, self._originalEvs[n])
    else:
      return self._estimators[i].value()

  def results(self, i):
    """
    return the results for node i
    :param i: the nodeId
    :return: a pair (gum.Potential,confidence)
    """
    if self._originalBN.variable(i).name() in self._originalEvs:
      return utils.deterministicPotential(self._originalBN.variable(i),
                                          self._originalEvs[self._originalBN.variable(i).name()]), 1.0

    else:
      return self._estimators[i].value(), self._estimators[i].confidence()
