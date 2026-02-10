"""
Weighted survey analysis utilities for SCF data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

class WeightedSurveyAnalyzer:
    """
    Handles weighted survey analysis for SCF data.
    """
    
    def __init__(self, data: pd.DataFrame, weight_col: str = 'WGT'):
        self.data = data
        self.weight_col = weight_col
        self.weights = data[weight_col] if weight_col in data.columns else None
        
    def weighted_mean(self, values: pd.Series) -> float:
        """
        Calculate weighted mean.
        """
        if self.weights is None:
            return values.mean()
        
        valid_mask = values.notna() & self.weights.notna()
        if valid_mask.sum() == 0:
            return np.nan
        
        return np.average(values[valid_mask], weights=self.weights[valid_mask])
    
    def weighted_median(self, values: pd.Series) -> float:
        """
        Calculate weighted median.
        """
        if self.weights is None:
            return values.median()
        
        valid_mask = values.notna() & self.weights.notna()
        if valid_mask.sum() == 0:
            return np.nan
        
        valid_values = values[valid_mask]
        valid_weights = self.weights[valid_mask]
        
        # Sort values and weights
        sorted_idx = valid_values.argsort()
        sorted_values = valid_values.iloc[sorted_idx]
        sorted_weights = valid_weights.iloc[sorted_idx]
        
        # Calculate cumulative weights
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights.iloc[-1]
        
        # Find median
        median_idx = np.searchsorted(cum_weights, total_weight / 2)
        
        return sorted_values.iloc[median_idx]
    
    def weighted_std(self, values: pd.Series) -> float:
        """
        Calculate weighted standard deviation.
        """
        if self.weights is None:
            return values.std()
        
        valid_mask = values.notna() & self.weights.notna()
        if valid_mask.sum() == 0:
            return np.nan
        
        valid_values = values[valid_mask]
        valid_weights = self.weights[valid_mask]
        
        weighted_mean = self.weighted_mean(values)
        variance = np.average((valid_values - weighted_mean)**2, weights=valid_weights)
        
        return np.sqrt(variance)
    
    def weighted_quantile(self, values: pd.Series, q: float) -> float:
        """
        Calculate weighted quantile.
        """
        if self.weights is None:
            return values.quantile(q)
        
        valid_mask = values.notna() & self.weights.notna()
        if valid_mask.sum() == 0:
            return np.nan
        
        valid_values = values[valid_mask]
        valid_weights = self.weights[valid_mask]
        
        # Sort values and weights
        sorted_idx = valid_values.argsort()
        sorted_values = valid_values.iloc[sorted_idx]
        sorted_weights = valid_weights.iloc[sorted_idx]
        
        # Calculate cumulative weights
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights.iloc[-1]
        
        # Find quantile
        quantile_idx = np.searchsorted(cum_weights, total_weight * q)
        
        return sorted_values.iloc[quantile_idx]
    
    def weighted_crosstab(self, var1: pd.Series, var2: pd.Series) -> pd.DataFrame:
        """
        Create weighted crosstabulation.
        """
        if self.weights is None:
            return pd.crosstab(var1, var2)
        
        # Create DataFrame with variables and weights
        df = pd.DataFrame({
            'var1': var1,
            'var2': var2,
            'weight': self.weights
        }).dropna()
        
        # Create weighted crosstab
        crosstab = df.pivot_table(
            values='weight',
            index='var1',
            columns='var2',
            aggfunc='sum',
            fill_value=0
        )
        
        return crosstab
    
    def weighted_groupby(self, group_var: pd.Series, 
                         target_var: pd.Series, 
                         agg_func: str = 'mean') -> pd.Series:
        """
        Perform weighted groupby operation.
        """
        if self.weights is None:
            if agg_func == 'mean':
                return target_var.groupby(group_var).mean()
            elif agg_func == 'median':
                return target_var.groupby(group_var).median()
            elif agg_func == 'std':
                return target_var.groupby(group_var).std()
            elif agg_func == 'count':
                return target_var.groupby(group_var).count()
        
        # Create DataFrame for weighted operations
        df = pd.DataFrame({
            'group': group_var,
            'target': target_var,
            'weight': self.weights
        }).dropna()
        
        if agg_func == 'mean':
            return df.groupby('group').apply(
                lambda x: np.average(x['target'], weights=x['weight'])
            )
        elif agg_func == 'median':
            return df.groupby('group').apply(
                lambda x: self._weighted_median_series(x['target'], x['weight'])
            )
        elif agg_func == 'std':
            return df.groupby('group').apply(
                lambda x: self._weighted_std_series(x['target'], x['weight'])
            )
        elif agg_func == 'count':
            return df.groupby('group')['weight'].sum()
    
    def _weighted_median_series(self, values: pd.Series, weights: pd.Series) -> float:
        """
        Helper function for weighted median in groupby.
        """
        sorted_idx = values.argsort()
        sorted_values = values.iloc[sorted_idx]
        sorted_weights = weights.iloc[sorted_idx]
        
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights.iloc[-1]
        
        median_idx = np.searchsorted(cum_weights, total_weight / 2)
        
        return sorted_values.iloc[median_idx]
    
    def _weighted_std_series(self, values: pd.Series, weights: pd.Series) -> float:
        """
        Helper function for weighted std in groupby.
        """
        weighted_mean = np.average(values, weights=weights)
        variance = np.average((values - weighted_mean)**2, weights=weights)
        
        return np.sqrt(variance)
    
    def weighted_correlation(self, var1: pd.Series, var2: pd.Series) -> float:
        """
        Calculate weighted correlation coefficient.
        """
        if self.weights is None:
            return var1.corr(var2)
        
        valid_mask = var1.notna() & var2.notna() & self.weights.notna()
        if valid_mask.sum() == 0:
            return np.nan
        
        valid_var1 = var1[valid_mask]
        valid_var2 = var2[valid_mask]
        valid_weights = self.weights[valid_mask]
        
        # Calculate weighted means
        mean1 = np.average(valid_var1, weights=valid_weights)
        mean2 = np.average(valid_var2, weights=valid_weights)
        
        # Calculate weighted covariance and variances
        cov = np.average((valid_var1 - mean1) * (valid_var2 - mean2), weights=valid_weights)
        var1 = np.average((valid_var1 - mean1)**2, weights=valid_weights)
        var2 = np.average((valid_var2 - mean2)**2, weights=valid_weights)
        
        # Calculate correlation
        if var1 == 0 or var2 == 0:
            return np.nan
        
        return cov / np.sqrt(var1 * var2)
    
    def create_weighted_quantiles(self, values: pd.Series, 
                                 n_quantiles: int) -> pd.Series:
        """
        Create weighted quantile categories.
        """
        if self.weights is None:
            return pd.qcut(values, n_quantiles, labels=False) + 1
        
        valid_mask = values.notna() & self.weights.notna()
        if valid_mask.sum() == 0:
            return pd.Series(np.nan, index=values.index)
        
        valid_values = values[valid_mask]
        valid_weights = self.weights[valid_mask]
        
        # Sort by values
        sorted_idx = valid_values.argsort()
        sorted_values = valid_values.iloc[sorted_idx]
        sorted_weights = valid_weights.iloc[sorted_idx]
        
        # Calculate cumulative weights
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights.iloc[-1]
        
        # Create quantile boundaries
        quantile_boundaries = np.linspace(0, total_weight, n_quantiles + 1)
        
        # Assign quantiles
        quantiles = pd.Series(np.nan, index=values.index)
        
        for i in range(n_quantiles):
            lower_bound = quantile_boundaries[i]
            upper_bound = quantile_boundaries[i + 1]
            
            if i == n_quantiles - 1:  # Last quantile includes upper bound
                mask = cum_weights >= lower_bound
            else:
                mask = (cum_weights >= lower_bound) & (cum_weights < upper_bound)
            
            quantile_indices = sorted_idx[mask]
            quantiles.iloc[quantile_indices] = i + 1
        
        return quantiles
    
    def weighted_describe(self, values: pd.Series) -> Dict[str, float]:
        """
        Generate weighted descriptive statistics.
        """
        if self.weights is None:
            return {
                'count': values.count(),
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                '25%': values.quantile(0.25),
                '50%': values.median(),
                '75%': values.quantile(0.75),
                'max': values.max()
            }
        
        valid_mask = values.notna() & self.weights.notna()
        if valid_mask.sum() == 0:
            return {stat: np.nan for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']}
        
        return {
            'count': valid_mask.sum(),
            'mean': self.weighted_mean(values),
            'std': self.weighted_std(values),
            'min': values.min(),
            '25%': self.weighted_quantile(values, 0.25),
            '50%': self.weighted_median(values),
            '75%': self.weighted_quantile(values, 0.75),
            'max': values.max()
        }