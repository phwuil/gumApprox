# -*- coding: utf-8 -*-
import pyAgrum as gum

import testUtils
from ApproxInference import Importance


def main():
  bn = gum.loadBN("data/alarm.bif")
  print("BN read")

  evs = {"HR": 1, "PAP": 2}

  m = Importance(bn, evs, verbose=True)
  m.run(5e-2, 50)
  print("done")

  testUtils.compareApprox(m, bn, evs)


if __name__ == '__main__':
  main()
