# Portfolio Optimisation with Modern Portfolio Theory  

This project explores portfolio construction using **Modern Portfolio Theory (MPT)**.  
It compares a simple **equal-weight portfolio** to optimised portfolios:  
- The **Minimum Variance Portfolio** (lowest risk)  
- The **Maximum Sharpe Portfolio** (best risk-adjusted return)  
- The **Efficient Frontier** (set of optimal portfolios for different risk levels)  

---

## 📊 Project Overview  
1. **Data Collection**  
   - Daily stock data pulled from Yahoo Finance (`yfinance`)  
   - Assets chosen from the **FTSE 100** across different sectors for diversification  

2. **Portfolio Construction**  
   - Equal-weight portfolio as a benchmark  
   - Optimised portfolios using `scipy.optimize.minimize` with constraints:
     - Weights must sum to 1 (fully invested)  
     - No short-selling (weights ≥ 0)  

3. **Analysis**  
   - Portfolio returns, risk (volatility), and Sharpe ratio  
   - Visualisation of the **Efficient Frontier** and key portfolios  

---

## 🔧 Requirements  
Install required packages using:  
```bash
pip install -r requirements.txt
