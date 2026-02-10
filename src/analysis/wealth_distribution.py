"""
Wealth Distribution and Inequality Analysis Module

This module analyzes wealth distribution patterns, inequality metrics,
and demographic disparities in the SCF 2022 dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

class WealthDistributionAnalyzer:
    """
    Analyzes wealth distribution and inequality patterns in SCF data.
    """
    
    def __init__(self, data: pd.DataFrame, weighted_analyzer):
        self.data = data
        self.analyzer = weighted_analyzer
        
    def analyze_wealth_distribution(self) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive wealth distribution analysis.
        """
        results = {}
        
        # Overall wealth distribution
        results['overall_distribution'] = self._overall_wealth_distribution()
        
        # Wealth by demographic groups
        results['by_demographics'] = self._wealth_by_demographics()
        
        # Wealth inequality metrics
        results['inequality_metrics'] = self._calculate_inequality_metrics()
        
        # Wealth mobility and concentration
        results['concentration_analysis'] = self._wealth_concentration_analysis()
        
        return results
    
    def _overall_wealth_distribution(self) -> pd.DataFrame:
        """
        Analyze overall wealth distribution patterns.
        """
        if 'NETWORTH' not in self.data.columns:
            return pd.DataFrame()
        
        # Create wealth percentiles
        wealth_values = self.data['NETWORTH']
        
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        distribution_data = []
        
        for p in percentiles:
            wealth_at_p = self.analyzer.weighted_quantile(wealth_values, p/100)
            distribution_data.append({
                'Percentile': p,
                'Wealth_Threshold': wealth_at_p,
                'Description': f'Wealth at {p}th percentile'
            })
        
        # Add summary statistics
        summary = self.analyzer.weighted_describe(wealth_values)
        distribution_data.extend([
            {'Percentile': 'Mean', 'Wealth_Threshold': summary['mean'], 'Description': 'Average wealth'},
            {'Percentile': 'Median', 'Wealth_Threshold': summary['50%'], 'Description': 'Median wealth'},
        ])
        
        return pd.DataFrame(distribution_data)
    
    def _wealth_by_demographics(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze wealth distribution by demographic groups.
        """
        results = {}
        
        # By age groups
        if 'AGE' in self.data.columns:
            results['by_age'] = self._wealth_by_age_groups()
        
        # By education
        if 'EDCL' in self.data.columns:
            results['by_education'] = self._wealth_by_education()
        
        # By race/ethnicity
        if 'RACECL' in self.data.columns:
            results['by_race'] = self._wealth_by_race()
        
        # By household type
        if 'MARRIED' in self.data.columns:
            results['by_household_type'] = self._wealth_by_household_type()
        
        return results
    
    def _wealth_by_age_groups(self) -> pd.DataFrame:
        """
        Analyze wealth distribution by age groups.
        """
        # Create age groups
        age_bins = [0, 35, 45, 55, 65, 100]
        age_labels = ['<35', '35-44', '45-54', '55-64', '65+']
        
        self.data['AGE_GROUP'] = pd.cut(self.data['AGE'], bins=age_bins, labels=age_labels, right=False)
        
        # Calculate wealth statistics by age group
        age_wealth = self.analyzer.weighted_groupby(
            self.data['AGE_GROUP'], 
            self.data['NETWORTH'], 
            'mean'
        ).round(0)
        
        age_median = self.analyzer.weighted_groupby(
            self.data['AGE_GROUP'], 
            self.data['NETWORTH'], 
            'median'
        ).round(0)
        
        age_count = self.analyzer.weighted_groupby(
            self.data['AGE_GROUP'], 
            self.data['NETWORTH'], 
            'count'
        ).round(0)
        
        result_df = pd.DataFrame({
            'Mean_Wealth': age_wealth,
            'Median_Wealth': age_median,
            'Weighted_Count': age_count
        }).reset_index()
        
        result_df.columns = ['Age_Group', 'Mean_Wealth', 'Median_Wealth', 'Weighted_Count']
        
        return result_df
    
    def _wealth_by_education(self) -> pd.DataFrame:
        """
        Analyze wealth distribution by education level.
        """
        education_labels = {
            1: 'Less than HS',
            2: 'HS diploma', 
            3: 'Some college',
            4: 'College degree',
            5: 'Postgraduate'
        }
        
        self.data['EDUCATION_LABEL'] = self.data['EDCL'].map(education_labels)
        
        # Calculate wealth statistics by education
        edu_wealth = self.analyzer.weighted_groupby(
            self.data['EDUCATION_LABEL'], 
            self.data['NETWORTH'], 
            'mean'
        ).round(0)
        
        edu_median = self.analyzer.weighted_groupby(
            self.data['EDUCATION_LABEL'], 
            self.data['NETWORTH'], 
            'median'
        ).round(0)
        
        edu_count = self.analyzer.weighted_groupby(
            self.data['EDUCATION_LABEL'], 
            self.data['NETWORTH'], 
            'count'
        ).round(0)
        
        result_df = pd.DataFrame({
            'Mean_Wealth': edu_wealth,
            'Median_Wealth': edu_median,
            'Weighted_Count': edu_count
        }).reset_index()
        
        result_df.columns = ['Education_Level', 'Mean_Wealth', 'Median_Wealth', 'Weighted_Count']
        
        return result_df
    
    def _wealth_by_race(self) -> pd.DataFrame:
        """
        Analyze wealth distribution by race/ethnicity.
        """
        race_labels = {
            1: 'White',
            2: 'Black',
            3: 'Hispanic',
            4: 'Asian',
            5: 'Other'
        }
        
        self.data['RACE_LABEL'] = self.data['RACECL'].map(race_labels)
        
        # Calculate wealth statistics by race
        race_wealth = self.analyzer.weighted_groupby(
            self.data['RACE_LABEL'], 
            self.data['NETWORTH'], 
            'mean'
        ).round(0)
        
        race_median = self.analyzer.weighted_groupby(
            self.data['RACE_LABEL'], 
            self.data['NETWORTH'], 
            'median'
        ).round(0)
        
        race_count = self.analyzer.weighted_groupby(
            self.data['RACE_LABEL'], 
            self.data['NETWORTH'], 
            'count'
        ).round(0)
        
        result_df = pd.DataFrame({
            'Mean_Wealth': race_wealth,
            'Median_Wealth': race_median,
            'Weighted_Count': race_count
        }).reset_index()
        
        result_df.columns = ['Race_Ethnicity', 'Mean_Wealth', 'Median_Wealth', 'Weighted_Count']
        
        return result_df
    
    def _wealth_by_household_type(self) -> pd.DataFrame:
        """
        Analyze wealth distribution by household type.
        """
        household_labels = {
            1: 'Married',
            2: 'Unmarried'
        }
        
        self.data['HOUSEHOLD_TYPE'] = self.data['MARRIED'].map(household_labels)
        
        # Calculate wealth statistics by household type
        hh_wealth = self.analyzer.weighted_groupby(
            self.data['HOUSEHOLD_TYPE'], 
            self.data['NETWORTH'], 
            'mean'
        ).round(0)
        
        hh_median = self.analyzer.weighted_groupby(
            self.data['HOUSEHOLD_TYPE'], 
            self.data['NETWORTH'], 
            'median'
        ).round(0)
        
        hh_count = self.analyzer.weighted_groupby(
            self.data['HOUSEHOLD_TYPE'], 
            self.data['NETWORTH'], 
            'count'
        ).round(0)
        
        result_df = pd.DataFrame({
            'Mean_Wealth': hh_wealth,
            'Median_Wealth': hh_median,
            'Weighted_Count': hh_count
        }).reset_index()
        
        result_df.columns = ['Household_Type', 'Mean_Wealth', 'Median_Wealth', 'Weighted_Count']
        
        return result_df
    
    def _calculate_inequality_metrics(self) -> pd.DataFrame:
        """
        Calculate wealth inequality metrics.
        """
        if 'NETWORTH' not in self.data.columns:
            return pd.DataFrame()
        
        wealth_values = self.data['NETWORTH'].dropna()
        
        # Gini coefficient
        gini = self._calculate_gini_coefficient(wealth_values)
        
        # Wealth share by top percentiles
        top_1_share = self._calculate_wealth_share(wealth_values, 0.99)
        top_5_share = self._calculate_wealth_share(wealth_values, 0.95)
        top_10_share = self._calculate_wealth_share(wealth_values, 0.90)
        top_20_share = self._calculate_wealth_share(wealth_values, 0.80)
        
        # Bottom 50 share
        bottom_50_share = self._calculate_wealth_share_bottom(wealth_values, 0.50)
        
        # Palma ratio (top 10% / bottom 40%)
        palma_ratio = top_10_share / (1 - top_10_share - bottom_50_share) if (1 - top_10_share - bottom_50_share) > 0 else np.nan
        
        metrics_data = [
            {'Metric': 'Gini_Coefficient', 'Value': gini, 'Description': 'Wealth inequality (0=equal, 1=unequal)'},
            {'Metric': 'Top_1_Percent_Share', 'Value': top_1_share, 'Description': 'Wealth share of top 1%'},
            {'Metric': 'Top_5_Percent_Share', 'Value': top_5_share, 'Description': 'Wealth share of top 5%'},
            {'Metric': 'Top_10_Percent_Share', 'Value': top_10_share, 'Description': 'Wealth share of top 10%'},
            {'Metric': 'Top_20_Percent_Share', 'Value': top_20_share, 'Description': 'Wealth share of top 20%'},
            {'Metric': 'Bottom_50_Percent_Share', 'Value': bottom_50_share, 'Description': 'Wealth share of bottom 50%'},
            {'Metric': 'Palma_Ratio', 'Value': palma_ratio, 'Description': 'Top 10% / Bottom 40% wealth ratio'},
        ]
        
        return pd.DataFrame(metrics_data)
    
    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """
        Calculate Gini coefficient for wealth distribution.
        """
        if self.analyzer.weights is not None:
            # Weighted Gini calculation
            valid_mask = values.notna() & self.analyzer.weights.notna()
            valid_values = values[valid_mask]
            valid_weights = self.analyzer.weights[valid_mask]
            
            # Sort by values
            sorted_idx = valid_values.argsort()
            sorted_values = valid_values.iloc[sorted_idx]
            sorted_weights = valid_weights.iloc[sorted_idx]
            
            # Calculate weighted Gini
            cum_weights = np.cumsum(sorted_weights)
            total_weight = cum_weights.iloc[-1]
            
            # Gini calculation
            n = len(sorted_values)
            cov = 0
            for i in range(n):
                for j in range(n):
                    cov += sorted_weights.iloc[i] * sorted_weights.iloc[j] * abs(sorted_values.iloc[i] - sorted_values.iloc[j])
            
            gini = cov / (2 * total_weight**2 * np.average(sorted_values, weights=sorted_weights))
            
        else:
            # Unweighted Gini calculation
            sorted_values = np.sort(values)
            n = len(values)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        
        return gini
    
    def _calculate_wealth_share(self, values: pd.Series, percentile: float) -> float:
        """
        Calculate wealth share above given percentile.
        """
        threshold = self.analyzer.weighted_quantile(values, percentile)
        
        if self.analyzer.weights is not None:
            top_mask = values >= threshold
            top_wealth = np.sum(values[top_mask] * self.analyzer.weights[top_mask])
            total_wealth = np.sum(values * self.analyzer.weights)
        else:
            top_wealth = values[values >= threshold].sum()
            total_wealth = values.sum()
        
        return top_wealth / total_wealth if total_wealth > 0 else 0
    
    def _calculate_wealth_share_bottom(self, values: pd.Series, percentile: float) -> float:
        """
        Calculate wealth share below given percentile.
        """
        threshold = self.analyzer.weighted_quantile(values, percentile)
        
        if self.analyzer.weights is not None:
            bottom_mask = values <= threshold
            bottom_wealth = np.sum(values[bottom_mask] * self.analyzer.weights[bottom_mask])
            total_wealth = np.sum(values * self.analyzer.weights)
        else:
            bottom_wealth = values[values <= threshold].sum()
            total_wealth = values.sum()
        
        return bottom_wealth / total_wealth if total_wealth > 0 else 0
    
    def _wealth_concentration_analysis(self) -> pd.DataFrame:
        """
        Analyze wealth concentration patterns.
        """
        if 'NETWORTH' not in self.data.columns:
            return pd.DataFrame()
        
        # Create wealth quintiles
        wealth_quintiles = self.analyzer.create_weighted_quantiles(self.data['NETWORTH'], 5)
        self.data['WEALTH_QUINTILE'] = wealth_quintiles
        
        # Calculate wealth share by quintile
        quintile_shares = []
        total_wealth = np.sum(self.data['NETWORTH'] * self.analyzer.weights) if self.analyzer.weights is not None else self.data['NETWORTH'].sum()
        
        for q in range(1, 6):
            quintile_mask = self.data['WEALTH_QUINTILE'] == q
            quintile_wealth = np.sum(self.data['NETWORTH'][quintile_mask] * self.analyzer.weights[quintile_mask]) if self.analyzer.weights is not None else self.data['NETWORTH'][quintile_mask].sum()
            
            share = quintile_wealth / total_wealth if total_wealth > 0 else 0
            
            quintile_shares.append({
                'Quintile': q,
                'Wealth_Share': share,
                'Cumulative_Share': sum([qs['Wealth_Share'] for qs in quintile_shares] + [share]),
                'Description': f'Wealth quintile {q} (20% of households)'
            })
        
        return pd.DataFrame(quintile_shares)
    
    def create_wealth_distribution_plots(self) -> Dict[str, plt.Figure]:
        """
        Create visualization plots for wealth distribution analysis.
        """
        plots = {}
        
        # Wealth distribution histogram
        plots['wealth_histogram'] = self._plot_wealth_histogram()
        
        # Wealth Lorenz curve
        plots['lorenz_curve'] = self._plot_lorenz_curve()
        
        # Wealth by demographics
        plots['demographic_comparison'] = self._plot_demographic_wealth_comparison()
        
        # Wealth concentration
        plots['concentration_chart'] = self._plot_wealth_concentration()
        
        return plots
    
    def _plot_wealth_histogram(self) -> plt.Figure:
        """
        Create wealth distribution histogram.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter for reasonable wealth range for visualization
        wealth_data = self.data['NETWORTH']
        wealth_filtered = wealth_data[(wealth_data >= -100000) & (wealth_data <= 5000000)]
        
        if self.analyzer.weights is not None:
            weights_filtered = self.analyzer.weights.loc[wealth_filtered.index]
            ax.hist(wealth_filtered, bins=50, weights=weights_filtered, alpha=0.7, color='steelblue')
        else:
            ax.hist(wealth_filtered, bins=50, alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Net Worth ($)')
        ax.set_ylabel('Number of Households (Weighted)')
        ax.set_title('Distribution of Household Net Worth')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
        
        plt.tight_layout()
        return fig
    
    def _plot_lorenz_curve(self) -> plt.Figure:
        """
        Create Lorenz curve for wealth distribution.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wealth_values = self.data['NETWORTH'].dropna()
        
        if self.analyzer.weights is not None:
            # Weighted Lorenz curve
            sorted_idx = wealth_values.argsort()
            sorted_wealth = wealth_values.iloc[sorted_idx]
            sorted_weights = self.analyzer.weights.iloc[sorted_idx]
            
            cum_wealth = np.cumsum(sorted_wealth * sorted_weights)
            cum_weights = np.cumsum(sorted_weights)
            
            # Normalize
            cum_wealth_norm = cum_wealth / cum_wealth.iloc[-1]
            cum_weights_norm = cum_weights / cum_weights.iloc[-1]
            
            ax.plot(cum_weights_norm, cum_wealth_norm, 'b-', linewidth=2, label='Actual Distribution')
        else:
            # Unweighted Lorenz curve
            sorted_wealth = np.sort(wealth_values)
            cum_wealth = np.cumsum(sorted_wealth)
            cum_wealth_norm = cum_wealth / cum_wealth[-1]
            cum_pop_norm = np.arange(1, len(wealth_values) + 1) / len(wealth_values)
            
            ax.plot(cum_pop_norm, cum_wealth_norm, 'b-', linewidth=2, label='Actual Distribution')
        
        # Line of equality
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Line of Equality')
        
        ax.set_xlabel('Cumulative Share of Households')
        ax.set_ylabel('Cumulative Share of Wealth')
        ax.set_title('Lorenz Curve - Wealth Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_demographic_wealth_comparison(self) -> plt.Figure:
        """
        Create demographic wealth comparison plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age groups
        if 'AGE_GROUP' in self.data.columns:
            age_wealth = self.analyzer.weighted_groupby(self.data['AGE_GROUP'], self.data['NETWORTH'], 'median')
            axes[0, 0].bar(age_wealth.index, age_wealth.values, color='skyblue')
            axes[0, 0].set_title('Median Wealth by Age Group')
            axes[0, 0].set_ylabel('Median Net Worth ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Education
        if 'EDUCATION_LABEL' in self.data.columns:
            edu_wealth = self.analyzer.weighted_groupby(self.data['EDUCATION_LABEL'], self.data['NETWORTH'], 'median')
            axes[0, 1].bar(edu_wealth.index, edu_wealth.values, color='lightgreen')
            axes[0, 1].set_title('Median Wealth by Education Level')
            axes[0, 1].set_ylabel('Median Net Worth ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Race
        if 'RACE_LABEL' in self.data.columns:
            race_wealth = self.analyzer.weighted_groupby(self.data['RACE_LABEL'], self.data['NETWORTH'], 'median')
            axes[1, 0].bar(race_wealth.index, race_wealth.values, color='salmon')
            axes[1, 0].set_title('Median Wealth by Race/Ethnicity')
            axes[1, 0].set_ylabel('Median Net Worth ($)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Household type
        if 'HOUSEHOLD_TYPE' in self.data.columns:
            hh_wealth = self.analyzer.weighted_groupby(self.data['HOUSEHOLD_TYPE'], self.data['NETWORTH'], 'median')
            axes[1, 1].bar(hh_wealth.index, hh_wealth.values, color='gold')
            axes[1, 1].set_title('Median Wealth by Household Type')
            axes[1, 1].set_ylabel('Median Net Worth ($)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Format y-axes
        for ax in axes.flat:
            if ax.get_ylabel().startswith('$'):
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
        
        plt.tight_layout()
        return fig
    
    def _plot_wealth_concentration(self) -> plt.Figure:
        """
        Create wealth concentration chart.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Wealth shares by percentile
        percentiles = [50, 80, 90, 95, 99]
        shares = []
        
        for p in percentiles:
            share = self._calculate_wealth_share(self.data['NETWORTH'], p/100)
            shares.append(share)
        
        ax1.bar(range(len(percentiles)), shares, color='darkblue')
        ax1.set_xlabel('Percentile')
        ax1.set_ylabel('Wealth Share')
        ax1.set_title('Wealth Concentration by Percentile')
        ax1.set_xticks(range(len(percentiles)))
        ax1.set_xticklabels([f'Top {100-p}%' for p in percentiles])
        ax1.grid(True, alpha=0.3)
        
        # Quintile shares
        if 'WEALTH_QUINTILE' in self.data.columns:
            quintile_data = self._wealth_concentration_analysis()
            ax2.pie(quintile_data['Wealth_Share'], 
                   labels=[f'Q{i}' for i in range(1, 6)],
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
            ax2.set_title('Wealth Share by Quintile')
        
        plt.tight_layout()
        return fig