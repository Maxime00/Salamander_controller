#!/usr/bin/env python3

"""This script discussess the basic math operations used in
Python."""

# Only necessary in Python 2

import cmc_pylog as pylog  # Import biolog module for log messages

pylog.info(3*'\t' + 20*'#' + ' MATH ' + 20*'#' + 3*'\n')

# Basic operations
pylog.info("Adding 132 + 123 : {}".format((132 + 123)))  # Add

pylog.info("Subrtacting 43 - 23 : {}".format((43 - 23)))  # Subtract

pylog.info("Multiply 65 * 87 : {}".format((65 * 87)))  # Multiply

pylog.info("Exponent 65**4 : {}".format((65**4)))  # Exponent

pylog.info("Modulo 5 % 4 : {}".format((5 % 4)))  # Modulo

pylog.info("Division 10 / 4 : {}".format((10/4)))  # Division

# True division (Python2)
pylog.info("Division 10.0 / 4 : {}".format((10.0/4)))

pylog.info("Integer division 10 // 4 : {}".format((10//4)))  # Integer division

