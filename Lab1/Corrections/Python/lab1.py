""" Lab 1 """

from exercise1 import exercise1
from exercise2 import exercise2


def main(clargs):
    """ Main """
    exercise1(clargs)
    exercise2(clargs)


if __name__ == "__main__":
    from cmcpack import parse_args
    CLARGS = parse_args()
    main(CLARGS)