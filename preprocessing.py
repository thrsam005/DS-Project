"""

######################### ATLAS Top Tagging Open Data ##########################

preprocessing.py - This script defines two functions. One applies a standard
pre-processing scheme to the constituent inputs, and the other standardizes the
high level quantity inputs. These functions are applied in script train.py

For a description of the pre-processing and resulting distributions, see
(*CDS pub note link*).

Author: Kevin Greif
Last updated 4/21/2022
Written in python 3

################################################################################

"""

# Numerical imports
import numpy as np

def high_level(data_dict):
    """ high_level - This function "standardizes" each of the high level
    quantities contained in data_dict (subtract off mean and divide by
    standard deviation).

    Arguments:
    data_dict (dict of np arrays) - The python dictionary containing all of
    the high level quantities. No naming conventions assumed.

    Returns:
    (array) - The high level quantities, stacked along the last dimension.
    """

    # Empty list to accept pre-processed high level quantities
    features = []

    # Loop through quantities in data dict
    for quant in data_dict.values():

        # Some high level quantities have large orders of magnitude. Can divide
        # off these large exponents before evaluating mean and standard
        # deviation
        if 1e5 < quant.max() <= 1e11:
            # Quantity on scale TeV (sqrt{d12}, sqrt{d23}, ECF1, Qw)
            quant /= 1e6
        elif 1e11 < quant.max() <= 1e17:
            # Quantity on scale TeV^2 (ECF2)
            quant /= 1e12
        elif quant.max() > 1e17:
            # Quantity on scale TeV^3 (ECF3)
            quant /= 1e18

        # Calculated mean and standard deviation
        mean = quant.mean()
        stddev = quant.std()

        # Standardize and append to list
        standard_quant = (quant - mean) / stddev
        features.append(standard_quant)

    # Stack quantities and return
    stacked_data = np.stack(features, axis=-1)

    return stacked_data
