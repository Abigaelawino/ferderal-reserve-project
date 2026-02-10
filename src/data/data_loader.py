"""
Data loading and cleaning utilities for SCF 2022 dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

class SCFDataLoader:
    """
    Handles loading, cleaning, and preprocessing of SCF 2022 data.
    """
    
    def __init__(self, data_path: str = "data/SCFP2022.csv"):
        self.data_path = data_path
        self.raw_data = None
        self.clean_data = None
        self.variable_definitions = self._load_variable_definitions()
    
    def _load_variable_definitions(self) -> Dict[str, str]:
        """
        Define variable descriptions based on SCF documentation.
        """
        return {
            # Demographics
            'YY1': 'Year of interview',
            'Y1': 'Year identifier',
            'WGT': 'Survey weight',
            'HHSEX': 'Head of household sex (1=Male, 2=Female)',
            'AGE': 'Age of head of household',
            'AGECL': 'Age category',
            'EDUC': 'Education level of head',
            'EDCL': 'Education category',
            'MARRIED': 'Marital status (1=Married, 2=Unmarried)',
            'KIDS': 'Number of children',
            'RACE': 'Race of head',
            'RACECL': 'Race category',
            
            # Income
            'INCOME': 'Total family income',
            'WAGEINC': 'Wages and salary income',
            'BUSSEFARMINC': 'Business/farm income',
            'INTDIVINC': 'Interest and dividend income',
            'KGINC': 'Capital gains income',
            'SSRETINC': 'Social security/retirement income',
            'TRANSFOTHINC': 'Transfer/other income',
            
            # Assets
            'ASSET': 'Total assets',
            'CHECKING': 'Checking accounts',
            'SAVING': 'Savings accounts',
            'STOCKS': 'Stocks and mutual funds',
            'RETQLIQ': 'Retirement accounts (liquid)',
            'HOUSES': 'Primary residence value',
            'VEHIC': 'Vehicle value',
            'BUS': 'Business equity',
            'OTHFIN': 'Other financial assets',
            
            # Debts
            'DEBT': 'Total debt',
            'MRTHEL': 'Mortgage debt',
            'CCBAL': 'Credit card debt',
            'VEH_INST': 'Vehicle installment debt',
            'EDN_INST': 'Education debt',
            'ODEBT': 'Other debt',
            
            # Net Worth
            'NETWORTH': 'Net worth (assets - debts)',
            
            # Financial Behavior
            'FINLIT': 'Financial literacy score',
            'SAVED': 'Amount saved last year',
            'SPENDMOR': 'Spent more than income (1=Yes, 0=No)',
            'SPENDLESS': 'Spent less than income (1=Yes, 0=No)',
            
            # Categories
            'NWCAT': 'Net worth category',
            'INCCAT': 'Income category',
            'ASSETCAT': 'Asset category',
        }
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw SCF data.
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.raw_data)} households with {len(self.raw_data.columns)} variables")
            return self.raw_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def clean_scf_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the SCF data.
        """
        if self.raw_data is None:
            self.load_data()
        
        df = self.raw_data.copy()
        
        # Handle missing values and invalid responses
        df = self._handle_missing_values(df)
        
        # Create derived variables
        df = self._create_derived_variables(df)
        
        # Apply data types
        df = self._apply_data_types(df)
        
        # Filter to valid observations
        df = self._filter_valid_observations(df)
        
        self.clean_data = df
        print(f"Cleaned dataset: {len(df)} households with {len(df.columns)} variables")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and invalid responses.
        """
        # Common missing value codes in SCF
        missing_codes = [-1, -2, -3, -4, -5, -6, -7, 999999, 999998, 999997]
        
        # Replace missing codes with NaN
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].replace(missing_codes, np.nan)
        
        return df
    
    def _create_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived variables for analysis.
        """
        # Wealth-to-income ratio
        if 'NETWORTH' in df.columns and 'INCOME' in df.columns:
            df['WEALTH_INCOME_RATIO'] = df['NETWORTH'] / df['INCOME']
            df['WEALTH_INCOME_RATIO'] = df['WEALTH_INCOME_RATIO'].replace([np.inf, -np.inf], np.nan)
        
        # Debt-to-income ratio
        if 'DEBT' in df.columns and 'INCOME' in df.columns:
            df['DEBT_INCOME_RATIO'] = df['DEBT'] / df['INCOME']
            df['DEBT_INCOME_RATIO'] = df['DEBT_INCOME_RATIO'].replace([np.inf, -np.inf], np.nan)
        
        # Asset composition ratios
        if 'ASSET' in df.columns:
            if 'HOUSES' in df.columns:
                df['HOUSING_RATIO'] = df['HOUSES'] / df['ASSET']
            if 'STOCKS' in df.columns:
                df['STOCK_RATIO'] = df['STOCKS'] / df['ASSET']
            if 'RETQLIQ' in df.columns:
                df['RETIREMENT_RATIO'] = df['RETQLIQ'] / df['ASSET']
        
        # Income composition ratios
        if 'INCOME' in df.columns:
            if 'WAGEINC' in df.columns:
                df['WAGE_RATIO'] = df['WAGEINC'] / df['INCOME']
            if 'BUSSEFARMINC' in df.columns:
                df['BUSINESS_RATIO'] = df['BUSSEFARMINC'] / df['INCOME']
            if 'INTDIVINC' in df.columns:
                df['INVESTMENT_RATIO'] = df['INTDIVINC'] / df['INCOME']
        
        return df
    
    def _apply_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply appropriate data types to variables.
        """
        # Categorical variables
        categorical_vars = ['HHSEX', 'AGECL', 'EDCL', 'MARRIED', 'RACECL', 'NWCAT', 'INCCAT', 'ASSETCAT']
        
        for var in categorical_vars:
            if var in df.columns:
                df[var] = df[var].astype('category')
        
        return df
    
    def _filter_valid_observations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to valid observations for analysis.
        """
        # Remove households with missing key variables
        key_vars = ['WGT', 'NETWORTH', 'INCOME', 'AGE']
        
        for var in key_vars:
            if var in df.columns:
                df = df[df[var].notna()]
        
        # Remove extreme outliers (likely data errors)
        if 'NETWORTH' in df.columns:
            # Filter negative net worth beyond reasonable limits
            df = df[df['NETWORTH'] > -10000000]  # -$10M threshold
        
        if 'INCOME' in df.columns:
            # Filter negative income
            df = df[df['INCOME'] >= 0]
        
        return df
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get weighted summary statistics for key variables.
        """
        if self.clean_data is None:
            self.clean_data()
        
        df = self.clean_data
        weights = df['WGT'] if 'WGT' in df.columns else None
        
        # Key variables for summary
        key_vars = ['NETWORTH', 'INCOME', 'ASSET', 'DEBT', 'AGE']
        available_vars = [var for var in key_vars if var in df.columns]
        
        summary = []
        
        for var in available_vars:
            if weights is not None:
                weighted_mean = np.average(df[var], weights=weights)
                weighted_std = np.sqrt(np.average((df[var] - weighted_mean)**2, weights=weights))
                weighted_median = self._weighted_median(df[var], weights)
            else:
                weighted_mean = df[var].mean()
                weighted_std = df[var].std()
                weighted_median = df[var].median()
            
            summary.append({
                'Variable': var,
                'Mean': weighted_mean,
                'Median': weighted_median,
                'Std': weighted_std,
                'Min': df[var].min(),
                'Max': df[var].max(),
                'Count': len(df)
            })
        
        return pd.DataFrame(summary)
    
    def _weighted_median(self, data: pd.Series, weights: pd.Series) -> float:
        """
        Calculate weighted median.
        """
        if weights is None:
            return data.median()
        
        sorted_data = data.sort_values()
        sorted_weights = weights.loc[data.sort_values().index]
        
        cumsum_weights = np.cumsum(sorted_weights)
        total_weight = cumsum_weights.iloc[-1]
        
        median_idx = np.searchsorted(cumsum_weights, total_weight / 2)
        
        return sorted_data.iloc[median_idx]
    
    def create_wealth_quintiles(self) -> pd.DataFrame:
        """
        Create wealth quintiles for analysis.
        """
        if self.clean_data is None:
            self.clean_data()
        
        df = self.clean_data.copy()
        
        # Create weighted wealth quintiles
        if 'NETWORTH' in df.columns and 'WGT' in df.columns:
            df['WEALTH_QUINTILE'] = self._create_weighted_quantiles(
                df['NETWORTH'], df['WGT'], 5, 'WEALTH_QUINTILE'
            )
        
        return df
    
    def _create_weighted_quantiles(self, data: pd.Series, weights: pd.Series, 
                                  n_quantiles: int, var_name: str) -> pd.Series:
        """
        Create weighted quantile categories.
        """
        # Sort by data values
        sorted_idx = data.argsort()
        sorted_data = data.iloc[sorted_idx]
        sorted_weights = weights.iloc[sorted_idx]
        
        # Calculate cumulative weights
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights.iloc[-1]
        
        # Create quantile boundaries
        quantile_boundaries = np.linspace(0, total_weight, n_quantiles + 1)
        
        # Assign quantiles
        quantiles = pd.Series(index=data.index, dtype=int)
        
        for i in range(n_quantiles):
            lower_bound = quantile_boundaries[i]
            upper_bound = quantile_boundaries[i + 1]
            
            mask = (cum_weights >= lower_bound) & (cum_weights < upper_bound)
            quantile_indices = sorted_idx[mask]
            quantiles.iloc[quantile_indices] = i + 1
        
        # Handle the last quantile (include upper bound)
        last_mask = cum_weights >= quantile_boundaries[-2]
        last_indices = sorted_idx[last_mask]
        quantiles.iloc[last_indices] = n_quantiles
        
        return quantiles