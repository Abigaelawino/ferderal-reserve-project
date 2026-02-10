# Federal Reserve Survey of Consumer Finances (SCF) Analysis

A comprehensive analysis of the 2022 Survey of Consumer Finances dataset, covering wealth distribution, income dynamics, asset ownership patterns, debt analysis, retirement readiness, and financial behavior across US households.

## Project Structure

```
ferderal-reserve-project/
├── data/                   # Raw and processed data
├── src/
│   ├── data/              # Data loading and cleaning
│   ├── analysis/         # Statistical analysis modules
│   ├── visualization/    # Plotting and dashboard
│   └── utils/            # Utility functions
├── output/
│   ├── figures/          # Generated plots
│   ├── tables/           # Summary tables
│   └── reports/          # Final reports
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Research Questions

1. **Wealth Distribution & Inequality** - How is wealth distributed across US households?
2. **Income Dynamics** - What are the primary income sources across different household types?
3. **Asset Ownership Patterns** - Which assets are most common across wealth quintiles?
4. **Debt & Financial Vulnerability** - What are the debt patterns across demographic groups?
5. **Retirement Readiness** - How prepared are different age cohorts for retirement?
6. **Financial Behavior & Literacy** - How does financial literacy correlate with financial outcomes?
7. **Housing Economics** - What factors influence homeownership rates and home equity?

## Key Features

- Weighted survey analysis for representative statistics
- Comprehensive demographic breakdowns
- Interactive visualizations and dashboards
- Reproducible research pipeline
- Detailed statistical analysis

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main analysis: `python src/main.py`
3. View results in `output/` directory

## Data Source

Federal Reserve Survey of Consumer Finances 2022
- 22,976 households
- 357 variables
- Nationally representative sample with survey weights