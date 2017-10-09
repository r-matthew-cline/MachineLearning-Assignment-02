###############################################################################
##
## Logistic Regression
##
## @author: Matthew Cline
## @version: 20171010
##
## Description: Logistic regression models mapping features of a flu survey
##              to the risk of contracting flu. Uses single variable models
##              as well as multivariable models. Also makes use of
##              regularization and feature scaling.
##
###############################################################################

import pandas as pd
import math
import matplotlib.pyplot as plt

####### IMPORT DATA FROM EXCEL FILE INTO PANDAS STRUCTURE #######
data = pd.read_excel('fluML.xlsx', sheetname='Sheet1', parse_cols=[2,9,16])
data.dropna(subset=['HndWshQual', 'Risk', 'Flu'], inplace=True)
print(data)
