import pandas as pd

df = pd.read_csv(r"C:\Users\Mokitha\OneDrive\Desktop\Data Science\Task 2\train.csv")
print("Data loaded successfully!")
print(df.head())

# Show basic info
print("\n--- Dataset Info ---")
print(df.info())

# Show missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())


# --- Data Cleaning ---

# Fill missing 'Age' with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop 'Cabin' column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Check missing values after cleaning
print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Survival count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Class distribution
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()


# Survival by gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by passenger class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age vs survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age Distribution by Survival")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("\n--- Key Insights ---")
print("- Females had higher survival rate.")
print("- Passengers in 1st class had higher survival rate.")
print("- Children and younger people were more likely to survive.")
print("- Strong negative correlation between Pclass and survival.")

df.to_csv("cleaned_titanic.csv", index=False)
print("Cleaned data saved as 'cleaned_titanic.csv'")


df.to_csv("cleaned_titanic.csv", index=False)
print(" Cleaned data saved as 'cleaned_titanic.csv'")

plt.savefig("plot_name.png")

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")  
plt.show()
