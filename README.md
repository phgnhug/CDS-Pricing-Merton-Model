# CDS Pricing with Merton Structural Credit Model

**Derivatives Markets 2400-QFU1DVM | University of Warsaw**  
**Team**: Nhung, Linn  
**Deadline**: 21 May 2026

---

## 🎯 Project Overview

This project implements **CDS (Credit Default Swap) pricing using the Merton structural credit model** — a cornerstone approach in quantitative finance for valuing credit derivatives.

### What We're Doing

We're building a **complete pipeline** that:

1. **Takes equity market data** (stock price, volatility, debt level)
2. **Calibrates a structural credit model** to infer unobservable asset volatility
3. **Computes default probabilities** from equity market signals
4. **Prices CDS contracts** using the structural framework
5. **Analyzes sensitivity** to market parameters
6. **Validates results** against real CDS market spreads

### Why This Matters

The Merton model elegantly connects equity markets (what we observe) to credit risk (what we want to know). This bridge is fundamental to pricing corporate debt and credit derivatives.

### Real-World Application

Investment banks use structural credit models like this to price CDS, assess corporate credit risk, and manage credit portfolios.

---

## 📊 The Model

### Merton Framework

**Core Idea**: A company's equity is a call option on its assets

**Calibration**: Extract asset volatility from equity market data using Newton-Raphson solver

**Default Probability**: Compute PD = N(-d2) where d2 is the Merton distance to default

**CDS Valuation**: CDS spread ≈ (1 − recovery) × PD × 10000 bps

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/phgnhug/CDS-Pricing-Merton-Model.git
cd CDS-Pricing-Merton-Model

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python main.py
```

---

## 📁 Project Structure
