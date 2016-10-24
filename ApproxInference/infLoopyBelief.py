# -*- coding: utf-8 -*-
import pyAgrum as gum

from . import utils
from .infGenericSampler import GenericInference


class MonteCarlo(GenericInference):

  def __init__(self, bn, evs, verbose=True):
    super().__init__(bn, evs, verbose)
