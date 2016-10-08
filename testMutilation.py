# -*- coding: utf-8 -*-

import pyAgrum as gum

from ApproxInference.utils import isAlmostEqualPot, mutilate


# This file to test utils.mutilate
def main():
  bn1 = gum.loadBN("alarm.bif")
  print("  - BN read")

  evs = {"SAO2": 2, "CATECHOL": 1}
  bn2 = mutilate(gum.BayesNet(bn1), evs)
  print("  - Mutilation done")

  ie1 = gum.LazyPropagation(bn1)
  ie1.setEvidence(evs)
  ie1.makeInference()

  ie2 = gum.LazyPropagation(bn2)
  ie2.setEvidence(evs)
  ie2.makeInference()
  print("  - Inference done")

  nb_errors = 0
  for n in bn1.names():
    if not isAlmostEqualPot(ie1.posterior(bn1.idFromName(n)), ie2.posterior(bn2.idFromName(n))):
      nb_errors += 1
      print("Error on {} : {} != {}".format(n,
                                            ie1.posterior(bn1.idFromName(n))[:],
                                            ie2.posterior(bn2.idFromName(n))[:]))
    else:
      pass
  if nb_errors > 0:
    print("Errors : {}".format(nb_errors))
  else:
    print("No error : inference results are identical.")


if __name__ == '__main__':
  main()
