import matplotlib.pyplot as plt
import pandas as pd


# m ==> model
def visualize():
    df = pd.read_csv("house_data.csv")

    y = df["Price"]
    # Visualization (EDA)
    plt.scatter(df["Size"], y)
    plt.title("Size vs Price")
    plt.xlabel("Size")
    plt.ylabel("Price")
    plt.show()

    plt.scatter(df["Bedrooms"], y)
    plt.title("Bedrooms vs Price")
    plt.xlabel("Bedrooms")
    plt.ylabel("Price")
    plt.show()

    plt.scatter(df["Age"], y)
    plt.title("Age vs Price")
    plt.xlabel("Age")
    plt.ylabel("Price")
    plt.show()
