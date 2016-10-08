# -*- coding: utf-8 -*-
import pyAgrum as gum

from ApproxInference import Gibbs
from ApproxInference import utils


def main():
  bn = gum.loadBN("data/alarm.bif")
  print("BN read")

  evs = {"HR": 1, "PAP": 2}

  m = Gibbs(bn, evs, verbose=True)
  m.run(5e-2, 20)

  print("done")

  ie = gum.LazyPropagation(bn)
  ie.setEvidence(evs)
  ie.makeInference()

  for i in bn.ids():
    v, c = m.results(i)
    if v is not None:
      print("{} : {:3.5f}\n    exact  : {}\n    approx : {}  ({:7.5f})".format(bn.variable(i).name(),
                                                                               utils.KL(ie.posterior(i), v),
                                                                               utils.compactPot(ie.posterior(i)),
                                                                               utils.compactPot(v),
                                                                               c))
    else:
      print("{}: {}".format(bn.variable(i).name(), evs[bn.variable(i).name()]))


if __name__ == '__main__':
  main()
