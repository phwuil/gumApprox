# -*- coding: utf-8 -*-

import pyAgrum as gum

from ApproxInference.utils import isAlmostEqualPot, conditionalModel
from ApproxInference import MonteCarlo


def test1(bn1):
  print("=====")
  print("TEST1")
  print("=====")

  evs = {"SAO2": 2, "CATECHOL": 1}
  print("EVIDENCES {}".format(evs))
  bn2, evs2 = conditionalModel(bn1, evs)
  print("  - Mutilation done")

  ie1 = gum.LazyPropagation(bn1)
  ie1.setEvidence(evs)
  ie1.makeInference()

  ie2 = gum.LazyPropagation(bn2)
  ie2.setEvidence(evs2)
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


def test2(bn1):
  print("=====")
  print("TEST2")
  print("=====")

  evs = {"HYPOVOLEMIA": 0, "CATECHOL": 1, "INTUBATION": 0}
  bn2, evs2 = conditionalModel(bn1, evs)

  # HYPOVOLEMIA has no parent
  if "HYPOVOLEMIA" in evs2:
    print("- HYPOVOLEMIA should not be in evs2")
  try:
    i = bn2.idFromName("HYPOVOLEMIA")
    print("- HYPOVOLEMIA should not be in bn2")
  except:
    pass

  # INTUBATION has no parent
  if "INTUBATION" in evs2:
    print("- INTUBATION should not be in evs2")
  try:
    i = bn2.idFromName("INTUBATION")
    print("- INTUBATION should not be in bn2")
  except:
    pass

  ie1 = gum.LazyPropagation(bn1)
  ie1.setEvidence(evs)
  ie1.makeInference()

  ie2 = gum.LazyPropagation(bn2)
  ie2.setEvidence(evs2)
  ie2.makeInference()
  print("  - Inference done")

  nb_errors = 0
  for n in bn2.names():
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


def test3(bn1):
  print("=====")
  print("TEST3")
  print("=====")

  evs = {"HYPOVOLEMIA": 0, "CATECHOL": 1, "INTUBATION": 0}
  m = MonteCarlo(bn1, evs)


# This file to test utils.conditionalModel
def main():
  bn1 = gum.loadBN("data/alarm.bif")
  print("  - BN read")
  print()
  test1(bn1)
  print()
  test2(bn1)
  print()
  test3(bn1)


if __name__ == '__main__':
  main()
