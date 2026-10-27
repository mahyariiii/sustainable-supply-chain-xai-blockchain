# A Blockchain and Explainable AI (XAI) Architecture for Sustainable Supply Chain Governance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Solidity](https://img.shields.io/badge/Solidity-0.8.24-blue)](https://soliditylang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)

This repository contains the official implementation, smart contracts, and experimental evaluation scripts for the research paper: **"A Blockchain and Explainable AI (XAI) Architecture for Sustainable Supply Chain Governance: Mitigating Oracle Vulnerabilities and Operational Risk in Complex Procurement"**.

## 📌 Abstract
Global supply networks face compounded disruptions driven by geopolitical instability and strict ESG mandates. While predictive machine learning models forecast supplier insolvency, their "black-box" nature impedes trust. Concurrently, blockchain technology provides immutability but suffers from the "blockchain oracle problem." This project introduces a hybrid socio-technical architecture uniting **Explainable AI (TreeSHAP & CQR)** with a **Byzantine Fault-Tolerant (BFT) multi-oracle consensus layer** and **Ethereum-compatible smart contracts**.

---

## 📂 Repository Structure
- `contracts/`: Contains the hardened Solidity smart contract (`ProcurementGovernance.sol`) compiled under version 0.8.24.
- `scripts/`: Python pipelines for LightGBM hyperparameter tuning via Optuna, Conformalized Quantile Regression (CQR), and SHAP attribution.
- `verification/`: Static security analysis audit logs generated via Slither.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js & Hardhat (for smart contract compilation and testing)

### 1. Python Environment Setup
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Smart Contract Testing (Hardhat)
\`\`\`bash
cd contracts
npx hardhat test
\`\`\`

---

## 📊 Empirical Benchmarks
The pipeline is evaluated on the open-source DataCo Global Supply Chain Dataset (180,519 transactions). 
- **Model Performance:** LightGBM achieves $F_1\text{-score} = 0.932$ and $\text{AUC-ROC} = 0.961$.
- **Smart Contract Gas Optimization:** Average deployment cost is ~1.48M gas, and risk ingestion costs ~127,800 gas.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
