import cmc_pylog as pylog
import datetime as dt
from math import sqrt, cos
import cmc_pylog
from math import sqrt
import math
"""This script introduces you to the useage of Imports in Python.
One of the most powerful tool of any programming langauge is to
be able to resuse code.
Python allows this by setting up modules. One can import existing
libraries using the import function."""

### IMPORTS  ###

pylog.info(3*'\t' + 20*'#' + ' IMPORTS ' + 20*'#' + 3*'\n')

# The command will crash the program since pylog is not a standard
# python knows and no module has been imported that defines it.
# To be able to run the program comment the line and run again.

# A generic import of a default module named math

# Now you have access to all the functionality availble
# in the math module to be used in this function
print(
    'Square root of 25 computed from math module : {}'.format(
        math.sqrt(25)))

# To import a specific function from a module
# Now you can avoid referencing that the sqrt function is from
# math module and directly use it.
print(
    """Square root of 25 computed from math module by importing only
    sqrt function: {}""".format(sqrt(25)))

# Import a user defined module
# Here we import biolog : Module developed to display log messages for the
# exercise

cmc_pylog.info('Module developed to display log messages for the exercises')
cmc_pylog.warning("""When you explicitly import functions from modules,
it can lead to naming errors!!!""")

# Importing multiple functions from the same module

# Defining an alias :
# Often having to reuse the actual name of module can be a pain.
# We can assign aliases to module names to avoid this problem

pylog.info("Here we import the module datetime as dt.")

# Getting to know the methods availble in a module
pylog.info(dir(math))

