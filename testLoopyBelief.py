# -*- coding: utf-8 -*-
import pyAgrum as gum

import testUtils
from ApproxInference import LoopyBeliefPropagation
from ApproxInference.utils import compactPot

def main():
  bn = gum.loadBN("data/alarm.bif")
  print("BN read")

  evs = {"HR": 1, "PAP": 2}

  m = LoopyBeliefPropagation(bn, evs, verbose=True)
  m.run(1e-5, 50)
  print("done")

  testUtils.compareApprox(m, bn, evs)


def test():
  bn = gum.fastBN("a->b->c;", 3)
  evs = {}

  m = LoopyBeliefPropagation(bn, evs, verbose=True)

  def traceMsg():
    print("\n\n")
    print("   {}->    {}->".format(compactPot(m._messages[0, 1]), compactPot(m._messages[1, 2])))
    print("a ------------------------------> b ------------------------------> c")
    print("   <-{}    <-{}".format(compactPot(m._messages[1, 0]), compactPot(m._messages[2, 1])))

  traceMsg()
  m.updateNodeMessages(2)
  traceMsg()

def testCedric():
  print("loading ...")
  bn=gum.loadBN("data/fichier_nan.bif")
  print("done")
  scenario = {'E7.ValueEE': 6}
  m = LoopyBeliefPropagation(bn, scenario, verbose=True)
  print("running ...")
  m.run(4e-1, 2)
  print("done")

if __name__ == '__main__':
  testCedric()
