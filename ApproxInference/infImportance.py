# -*- coding: utf-8 -*-
import pyAgrum as gum
from pyAgrum.lib.bn2graph import pdfize

from . import utils
from .infGenericSampler import GenericSamplerInference
from .probabilityEstimator import ProbabilityEstimator


class Importance(GenericSamplerInference):
  def __init__(self, bn, evs, epsilon, verbose=True):
    """
    Importance sampling on bn with evs using the sampling distribution samplerBN

    :param bn: a bayesian network
    :param evs: map of evidence
    :param epsilon: the value used to 'unsharpen' the BN
    :param verbose:
    """
    super().__init__(bn, evs, verbose)

    self._samplerBN = utils.mutilatedModel(self._bn, self._evs)
    if self._verbose:
      print("Mutilating sampler distribution")
      print("  - dimension from {} to {}".format(self._bn.dim(), self._samplerBN.dim()))

    minParam = self._samplerBN.minNonZeroParam()
    minAccepted = epsilon / bn.maxVarDomainSize()
    if minParam < minAccepted:
      utils.unsharpenedModel(self._samplerBN, minAccepted)
      if self._verbose:
        print("Minimum parameter : {} => unsharpening sampler distribution to {}".format(minParam, minAccepted))
    pdfize(self._samplerBN, "test2.pdf")

    self._estimators = {i: ProbabilityEstimator(self._samplerBN.variable(i)) for i in self._samplerBN.ids() if
                        self._samplerBN.variable(i).name() not in evs}
    self._nbrReject = 0
    self._nbr = 0

  def multipleRound(self, size=1000):
    estimators = {i: ProbabilityEstimator(self._samplerBN.variable(i))
                  for i in self._samplerBN.ids()
                  if self._samplerBN.variable(i).name() not in self._evs}

    def oneRound():
      self._nbr += 1
      currentPotentials = {}
      proba = {}

      # simple sampling w.r.t. the samplerBN
      inst = dict(self._originalEvs)
      probaQ = 1.0
      probaP = 1.0
      for i in self._samplerBN.topologicalOrder():
        name = self._samplerBN.variable(i).name()
        q = gum.Potential(self._samplerBN.cpt(i))
        for j in self._samplerBN.parents(i):
          q *= currentPotentials[j]
        q = q.margSumIn([name])
        inst[name], currentPotentials[i] = utils.draw(q)
        if name not in self._evs:
          proba[i] = q
        probaQ *= self._samplerBN.cpt(name)[inst]
        probaP *= self._bn.cpt(name)[inst]
      for name in self._bn.names():
        if name not in self._samplerBN.names():  # variable that have disappeared in sampler_bn
          probaP *= self._bn.cpt(name)[inst]

      if probaP == 0:
        return False  # rejet

      for i in proba.keys():
        estimators[i].add(currentPotentials[i], probaP / probaQ)
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
