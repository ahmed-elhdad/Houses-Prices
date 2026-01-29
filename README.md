# 🏠 House Price Prediction using Machine Learning

> A professional Machine Learning project that predicts house prices based on size, number of bedrooms, and house age using regression models.

---

## 📌 Problem Statement

Estimating house prices accurately is critical for:
- Real estate companies
- Buyers and sellers
- Market analysis

This project aims to build a regression model that predicts house prices using historical data.

---

## 📊 Dataset

The dataset contains the following features:

| Feature | Description |
|------|------------|
| Size | House size in square feet |
| Bedrooms | Number of bedrooms |
| Age | Age of the house (years) |
| Price | House price (in thousands of dollars) |

---

## 🧠 Machine Learning Approach

1. Load and explore the dataset  
2. Split data into **training and testing sets**  
3. Scale features using **StandardScaler**  
4. Generate **Polynomial Features**  
5. Train a **Ridge Regression** model  
6. Evaluate performance using regression metrics  

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
https://github.com/ahmed-elhdad/Houses-Prices.git
cd Houses-Prices
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the training script
```bash
cd src
python train.py
```
## 🛠 Technologies Used
- Python

- pandas

- matplotlib

- scikit-learn
## 📈 Model Evaluation Metrics
- R² Score
- MAE
- MSE

<hr/>

## 📊 Visual Analysis

<h2>Scatter plots are used to analyze relationships between:</h2>

- Size vs Price

- Bedrooms vs Price

- Age vs Price
<hr/>

## 🚀 Future Improvements

- Increase dataset size

- Try Lasso and ElasticNet

- Perform hyperparameter tuning

- Save trained model for deployment

- Build a web interface
<hr/>

## 👨‍💻 Author

### Ahmed Alhdad
<p>Machine Learning & Python Developer</p>

<hr/>

## ⭐ If you find this project useful, feel free to star the repository!
