#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:46:53 2019

@author: mulugetasemework
"""
import numpy as np
Zscore = 1.28 # 95% confidence level, 80%	85%	90% 95% 99% 1.28  1.44	1.645 1.96 2.576



stdv = 0.5 # expected variance, the (estimated) proportion of the population

#     which has the attribute in question, we assume half have it
MErr = 0.1 #+/- 5 CI, the desired level of precision (i.e. the margin of error),

#    Cochran’s Formula Example
#    Suppose we are doing a study on the inhabitants of a large town, and want to 
#    find out how many households serve breakfast in the mornings. We don’t 
#    have much information on the subject to begin with, so we’re going to assume
#     that half of the families serve breakfast: this gives us maximum variability.
#     So p = 0.5. Now let’s say we want 95% confidence, and at least 5 percent—plus
#     or minus—precision. A 95 % confidence level gives us Z values of 1.96, 
#     per the normal tables, so we get
#  
#    ((1.96)2 (0.5) (0.5)) / (0.05)2 = 385.
Needed_Sample_Size = round((np.square(Zscore) * (stdv*(1-stdv)))/ np.square(MErr))

print(int(Needed_Sample_Size))
