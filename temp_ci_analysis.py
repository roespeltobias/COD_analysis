#Importing modules
import pandas as pd
import numpy as np
from typing import Tuple

#In particular for CausalImpact
from causalimpact import CausalImpact

def time_casual_impact_analysis_one_pt(epi_data: pd.array, response_data: pd.array, change_points: np.array, results_out: bool = True, debug_out: bool = False, change_pt: int = 0) -> object:
    """Calculates the causal impact of a response variable on an epidemiological variable using the CausalImpact package for one change point.

    Args:
        epi_data (np.array): Epidemiological data.
        response_data (np.array): Response data.
        change_points (np.array): Array containing the change points.
        results_out (bool): If True, print results. If False, do not print.
        debug_out (bool): If True, print debug information. If False, do not print.
        change_pt (int): The specific change point to use.
        
    Returns:
        object: Causal Impact Object
    """
    if change_pt == 0:
        y = epi_data[:change_points[1]+1].values
        X = response_data[:change_points[1]+1].values+np.random.normal(0, 0.1, len(y)) #Add noise to avoid zero variance in values which would cause inf problem in CausalImpact
        t = np.arange(len(y), dtype=int)
        X = t + X * t.mean()
        pre_period = [0, int(change_points[0])]
        post_period = [int(change_points[0]+1), int(change_points[1])]
    else:
        y = epi_data[change_points[change_pt-1]+1:change_points[change_pt+1]+1].values
        X = response_data[change_points[change_pt-1]+1:change_points[change_pt+1]+1].values+np.random.normal(0, 0.1, len(y)) #Add noise to avoid zero variance in values which would cause inf problem in CausalImpact
        t = np.arange(len(y), dtype=int)
        X = t + X * t.mean()
        pre_period = [0, int(change_points[change_pt])-int(change_points[change_pt-1])]
        post_period = [int(change_points[change_pt])-int(change_points[change_pt-1])+1, int(change_points[change_pt+1])-int(change_points[change_pt-1]) -1]
        
    if debug_out:
        print("Nans in y:", np.isnan(y).sum())
        print("Nans in X:", np.isnan(X).sum())
        print("Inf in y:", np.isinf(y).sum())
        print("Inf in X:", np.isinf(X).sum())
        print("Periods:", pre_period, post_period)
        
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])

    ci = CausalImpact(data, pre_period, post_period)
    if results_out:
        print(ci.summary())
        print(ci.summary(output='report'))
        ci.plot()
    return ci

def time_causal_impact_analysis(epi_data: pd.array, response_data: pd.array, all_change_pts: bool = True, results_out: bool = True, debug_out: bool = False, change_pt: int = 0) -> Tuple:
    """
    Perform a Causal Impact analysis on the given data.
    
    Parameters:
        epi_data (np.array): Epidemiological data.
        response_data (np.array): Response data.
        all_change_pts (bool): If True, use all change points. If False, use only the specified change point.
        results_out (bool): If True, print results. If False, do not print.
        debug_out (bool): If True, print debug information. If False, do not print.
        change_pt (int): The specific change point to use if all_change_pts is False.
        
    Returns:
        Tuple: A tuple containing the change points and the Causal Impact objects.
    """
    
    #Determine change points
    change_points = np.where(np.diff(response_data) != 0)[0]
    if results_out:
        print("Change points:", change_points)
    
    #Perform Causal Impact analysis
    if all_change_pts:
        ci_objects = []
        for i in range(len(change_points)-1):
            ci_objects.append(time_casual_impact_analysis_one_pt(epi_data, response_data, change_points, results_out, debug_out, i))
    else:
        ci_objects = time_casual_impact_analysis_one_pt(epi_data, response_data, change_points, results_out, debug_out, change_pt)
    return change_points, ci_objects