# -*- coding: utf-8 -*-

import pyAgrum as gum

from ApproxInference import MonteCarlo


def multipleMontecarlo(bn, evs,N=5):
  for i in range(1, N + 1):
    print()
    print("==> montecarlo {}/{}".format(i, N))
    m = MonteCarlo(bn,evs)
    m.run(1e-1, 50, verbose=True)

    for n in ["P3.Duration", "R3.Cost2", "A3.Productivity1"]:
      i = bn.idFromName(n)
      v, c = m.results(i)
      print("{}: {} ({})".format(bn.variable(i).name(), utils.compactPot(v), c))


def transform():
  bn1 = gum.loadBN("data/test_level_0.o3prm", system="aSys")
  print("transform : prm loaded")

  gum.saveBN(bn1, "data/test_level_0.bif")
  print("transform : bn written")

  bn = gum.loadBN("data/test_level_0.bif")
  print("transform : bn loaded")

  for i in bn.ids():
    bn.cpt(i).translate(1e-2).normalizeAsCPT()
  print("transform : bn normalized")

  gum.saveBN(bn, "data/test_level_0_1.bif")
  print("transform : bn written")


if __name__ == '__main__':
  transform()
  bn = gum.loadBN("data/test_level_0_1.bif")
  print("BN loaded")
  multipleMontecarlo(bn,evs={})
