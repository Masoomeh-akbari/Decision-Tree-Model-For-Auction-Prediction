# Decision-Tree-Model-For-Auction-Prediction

## Repository Contents
This repository includes:
- **R Script**: [`eBay.R`](./eBay.R) - Code for data preprocessing, model training, and visualization.
- **PDF Report**: [`eBay.pdf`](./eBay.pdf) - Detailed documentation of the methodology, analysis, and results.
- **Dataset**: [`eBayAuctions.csv`](./eBayAuctions.csv) - Contains auction data used in this analysis.

## Overview
This project focuses on building a classification model to predict whether eBay auctions are **competitive** (with at least two bids) or **noncompetitive**. Using data from 1972 auctions conducted during May-June 2004, the analysis examines how features like item category, seller rating, auction duration, and pricing influence auction outcomes.

## Methodology
- **Data Preprocessing**: Converted `Duration` into a categorical variable and split the dataset into training (60%) and validation (40%) subsets.
- **Modeling**: Fitted a pruned classification tree with constraints to avoid overfitting (minimum terminal node size = 50, max depth = 7).
- **Simplification**: Identified and removed less effective variables to improve accuracy and clarity.
- **Visualization**: Generated scatter plots to illustrate decision boundaries based on key predictors.

## Results
- **Validation Accuracy**: The pruned classification tree achieved a validation accuracy of **81.37%**, demonstrating reliable predictive performance.
- **Key Predictors**:
  - `OpenPrice`: Auctions with higher opening prices were more likely to be competitive.
  - `ClosePrice`: Final auction prices significantly impacted competitiveness, with higher values correlating to competitive auctions. However, `ClosePrice` presents challenges for real-time prediction, as it is only available after the auction concludes.
- **Simplified Model**: By removing less impactful variables such as `Category`, the model became more interpretable and achieved better performance.
- **Decision Rules**: Clear rules were derived from the tree, such as thresholds for `OpenPrice` and `ClosePrice`, offering actionable insights for sellers.

The analysis highlights the importance of pricing strategies in influencing auction competitiveness. However, reliance on `ClosePrice` limits the model's utility for predicting outcomes of new auctions in real-time. For detailed decision rules and additional insights, refer to [`eBay.R`](./eBay.R) and [`eBay.pdf`](./eBay.pdf).
