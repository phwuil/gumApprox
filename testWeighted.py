# -*- coding: utf-8 -*-
import pyAgrum as gum

import testUtils
from ApproxInference import Weighted


def main():
  bn = gum.loadBN("data/alarm.bif")
  print("BN read")

  evs = {"HR": 1, "PAP": 2}

  m = Weighted(bn, evs, verbose=True)
  m.run(5e-3, 50)
  print("done")

  testUtils.compareApprox(m, bn, evs)


if __name__ == '__main__':
  main()
