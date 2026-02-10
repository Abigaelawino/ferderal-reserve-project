# Studio 4 Project: Education, Wealth, and Financial Stability

**Research Question**: To what extent do pre-existing household wealth and demographic characteristics moderate the predictive relationship between higher education attainment and long-term financial stability for households within the same income quintile?

## Project Structure

```
studio-4-project/
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Python modules
├── output/
│   ├── figures/          # Generated plots
│   ├── tables/           # Summary tables
│   └── reports/          # Final reports
└── README.md
```

## Research Framework

### Target Variables
- **Payment Stress**: LATE, LATE60 (late debt payments)
- **Debt Burden**: DEBT2INC, PIR40 (debt-to-income ratios)
- **Financial Position**: NETWORTH (household net worth)
- **Financial Knowledge**: KNOWL (personal finance knowledge)

### Predictor Variables
- **Education**: EDCL, EDUC (education attainment)
- **Income Controls**: INCOME, INCCAT, INCPCTLECAT
- **Demographics**: RACECL4, HHSEX, AGE, MARRIED, KIDS
- **Wealth Background**: NETWORTH, NWPCTLECAT
- **Debt Structure**: HEDN_INST, EDN_INST, DEBT, CCBAL, RESDBT

### Interaction Effects
- **edcl × racecl4**: Education benefits by race group
- **edcl × nwpctlecat**: Education benefits by wealth background
- **edcl × income**: Education benefits by income level

### Financial Stability Index (FSI)
**Components**:
- **Payment Stress**: LATE, LATE60, PIR40
- **Debt Burden**: DEBT2INC, PIRTOTAL, LEVRATIO
- **Financial Resilience**: NETWORTH, LIQ, SAVED

## Analysis Plan

1. **Research Setup**: Load data, create variables, validate methodology
2. **Descriptive Analysis**: Explore relationships and patterns
3. **Regression Analysis**: Test main effects and interactions
4. **FSI Development**: Create and validate Financial Stability Index
5. **Results Synthesis**: Generate final report and visualizations

## Data Source

Federal Reserve Survey of Consumer Finances 2022 (SCF2022)
- 22,976 households
- Weighted survey analysis for representative results
- Focus on within-income-quintile analysis

## Expected Outcomes

- Quantified education effects on financial stability
- Identified moderation patterns by demographics and wealth
- Validated Financial Stability Index
- Policy-relevant insights for education and financial planning

---
**Status**: Ready for implementation
**Dependencies**: MVP Notebooks 00-02 completed