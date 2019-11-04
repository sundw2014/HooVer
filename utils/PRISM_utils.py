def And(*args):
    expr = ' & '.join(['('+arg+')' for arg in args])
    return expr

def Or(*args):
    expr = ' | '.join(['('+arg+')' for arg in args])
    return expr

class PrismExpression(object):
    """docstring for ."""
    def __init__(self, repr):
        super(PrismExpression, self).__init__()
        self.repr = repr

    def __repr__(self):
        return self.repr

    def __sub__(self, other):
        return PrismExpression('(' + self.repr + ') - (' + repr(other) + ')')

    def __add__(self, other):
        return PrismExpression('(' + self.repr + ') + (' + repr(other) + ')')

    def __gt__(self, other):
        return PrismExpression('(' + self.repr + ') > (' + repr(other) + ')')

    def __lt__(self, other):
        return PrismExpression('(' + self.repr + ') < (' + repr(other) + ')')
