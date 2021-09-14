""" Lab 1 """

from exercise3 import exercise3
from exercise4 import exercise4


def main(clargs):
    """ Main """
    exercise3(clargs)
    exercise4(clargs)


if __name__ == "__main__":
    from cmcpack import parse_args
    CLARGS = parse_args()
    main(CLARGS)