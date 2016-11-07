# -*- coding: utf-8 -*-
import random

import pyAgrum as gum

from . import utils
from .infGenericSampler import GenericInference


class LoopyBeliefPropagation(GenericInference):
  def __init__(self, bn, evs, verbose=True):
    super().__init__(bn, evs, verbose)
    self._messages = {}
    for a, b in self._bn.arcs():
      self._messages[a, b] = gum.Potential().add(self._bn.variable(a)).fillWith(1).normalize()
      self._messages[b, a] = gum.Potential().add(self._bn.variable(a)).fillWith(1)
    for a in self._evs:
      print("evidence {}:{}".format(a, evs[a]))
      nid = self._bn.idFromName(a)
      self._messages[-1, nid] = utils.deterministicPotential(self._bn.variable(nid), evs[a])

    for i in self._bn.topologicalOrder():
      self.updateNodeMessages(i)

    # we need a list of ids we can shuffle
    self._shuffledIds = list(self._bn.ids())

  def _computeProdPi(self, nodeX, nodeExcept=None):
    varX = self._bn.variable(nodeX)
    piX = gum.Potential(self._bn.cpt(nodeX))
    for par in self._bn.parents(nodeX):
      if par != nodeExcept:
        piX = piX * self._messages[par, nodeX]

    return piX

  def _computeProdLambda(self, nodeX, nodeExcept=None):
    varX = self._bn.variable(nodeX)
    if (-1, nodeX) in self._messages:
      lamX = gum.Potential(self._messages[-1, nodeX])
    else:
      lamX = gum.Potential().add(varX).fillWith(1.0)
    for child in self._bn.children(nodeX):
      if child != nodeExcept:
        lamX = lamX * self._messages[child, nodeX]
    return lamX

  def updateNodeMessages(self, nodeX):
    """
    compute all the messages from nodeX. Return the max difference between old and new messages and argmax arc (x,y)

    :param nodeX: the nodeId
    :return: KL,(x,y)
    """
    varX = self._bn.variable(nodeX)

    piX = self._computeProdPi(nodeX)
    piX = piX.margSumIn([varX.name()])

    lamX = self._computeProdLambda(nodeX)

    KL = -1.0
    argKL = (-1, -1)

    # update lambda_par (for arc par->x)
    for par in self._bn.parents(nodeX):
      newLambda = self._computeProdPi(nodeX, par)
      newLambda = newLambda.margSumIn([varX.name(), self._bn.variable(par).name()]) * lamX
      newLambda = newLambda.margSumOut([varX.name()])

      diff = utils.KL(newLambda, self._messages[nodeX, par])
      if diff > KL:
        KL = diff
        argKL = (nodeX, par)

      self._messages[nodeX, par] = newLambda

    # update pi_child (for arc x->child)
    for child in self._bn.children(nodeX):
      # lamX except lam(chi)
      newPi = (piX * self._computeProdLambda(nodeX, child)).normalize()

      diff = utils.KL(newPi, self._messages[nodeX, child])
      if diff > KL:
        KL = diff
        argKL = (nodeX, child)

      self._messages[nodeX, child] = newPi

    # return the KL max
    return KL, argKL

  def updateMessages(self):
    """
    update all the messages of the self._bn by iterating over all the nodes in random order.
    Return the max difference between old and new messages and argmax arc (x,y)

    :return: KL,(x,y)
    """
    random.shuffle(self._shuffledIds)

    maxKL = -1
    argMaxKL = (-1, -1)

    for nid in self._shuffledIds:
      kl, amkl = self.updateNodeMessages(nid)

      if kl > maxKL:
        maxKL = kl
        argMaxKL = amkl

    return maxKL, argMaxKL

  def run(self, epsilon, niter):
    while True:
      maxKL = -1
      argMaxKL = (-1, -1)
      for i in range(niter):

        kl, amkl = self.updateMessages()

        if kl > maxKL:
          maxKL = kl
          argMaxKL = amkl

      if self._verbose:
        print("KL={:0.6f} on {}-{}".format(maxKL,
                                           self._bn.variable(argMaxKL[0]).name(),
                                           self._bn.variable(argMaxKL[1]).name()))
      if maxKL < epsilon:
        return maxKL, argMaxKL

  def posterior(self, nodeX):
    p = self._computeProdPi(nodeX).margSumIn([self._bn.variable(nodeX).name()])
    p = p * self._computeProdLambda(nodeX)
    p.normalize()

    return p
