from minics import NonlinearProblem, ParameterContinuation
from dolfin import IntervalMesh, Constant


class Bratu(NonlinearProblem):

    def mesh(self):
        return IntervalMesh(1000, 0, 1)

    def parameters(self):
        return {"lambda": Constant(0)}


if __name__ == '__main__':
    bratu = Bratu()
    analysis = ParameterContinuation(bratu, "lambda")
