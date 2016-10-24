# -*- coding: utf-8 -*-
import pyAgrum as gum

from . import utils
from .infGenericSampler import GenericInference


class LoopyBeliefPropagation(GenericInference):
  def __init__(self, bn, evs, verbose=True):
    super().__init__(bn, evs, verbose)
    self._messages = {}
    for a, b in self._bn.arcs():
      self._messages[a, b] = gum.Potential().add(self._bn.variable(a))

  def run(self, epsilon, niter):
    for i in
