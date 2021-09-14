#!/usr/bin/env python3

"""This script explains the basic data types used in Python.
All the comman data types used in programming langauges are in Python too.
"""

# Import the biolog module to display log messages
import cmc_pylog as pylog

pylog.info(3*'\t' + 20*'#' + ' DATA TYPES ' + 20*'#' + 3*'\n')

pylog.warning(
    "In Python every element is treated as an Object."
    + " Including numbers and literals!!!"
)

# Different Data types
# Use 'type' method to identify the types
pylog.info('Data Type of 2 is : {}'.format(type(2)))  # int

pylog.info('Data Type of 2.0 is : {}'.format(type(2.0)))  # Float

pylog.info('Data Type of \'two\' is : {}'.format(type('two')))  # String

# Strings in can be three types:
# 1. '' -> single line strings
# 2. "" -> single line strings
# 3. """ """ -> Multi line strings

pylog.warning('There is no separate char data type in Python.')

# Boolean
pylog.info('Data Type of keyword True is : {}'.format(type(True)))

# None type
pylog.info('Data Type of keyword None is : {}'.format(
    type(None)))

