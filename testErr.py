import sys

sys.path.insert(0, '/home/phw/virtualenvs/devAgrum/lib/python3.5/site-packages')

import pyAgrum as gum
from pyAgrum.lib.bn2graph import pdfize

from ApproxInference import Importance, MonteCarlo

bn_test = gum.loadBN("data/errApprox.bif")
pdfize(bn_test, "test.pdf")

scenario = {'FA.Duration': 6}

ie = gum.LazyPropagation(bn_test)
ie.setEvidence(scenario)
ie.makeInference()

"""
ieLB = gum.LoopyBeliefPropagation(bn_test)
ieLB.setEvidence(scenario)
ieLB.makeInference()

ieG = gum.GibbsInference(bn_test)
ieG.setEvidence(scenario)
ieG.makeInference()

ieM = MonteCarlo(bn_test, scenario)
ieM.run(3e-2, 50)

"""
ieI = Importance(bn_test, scenario, epsilon=0.1, verbose=True)
ieI.run(1e-2, 100)

print('Lazy', ie.posterior(bn_test.idFromName("O.Productivity")))
"""
print('Loopy', ieLB.posterior(bn_test.idFromName("O.Productivity")))
print('Gibbs', ieG.posterior(bn_test.idFromName("O.Productivity")))
print('Montecarlo', ieM.posterior(bn_test.idFromName("O.Productivity")))
"""

print('Importance', ieI.posterior(bn_test.idFromName("O.Productivity")))
