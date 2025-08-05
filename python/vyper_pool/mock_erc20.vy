# pragma version 0.4.3

from snekmate.tokens import erc20
from snekmate.auth import ownable

initializes: ownable
initializes: erc20[ownable := ownable]

exports: erc20.__interface__


@deploy
def __init__(name: String[25], symbol: String[5]):
    ownable.__init__()
    erc20.__init__(name, symbol, 18, name, "1.0.0")