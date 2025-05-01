# Data Visualization I -1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about the passengers 
# who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we can find any patterns in the data. 
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by plotting a histogram.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df=pd.read_csv("titanic.csv")  # Ensure the dataset is in the same directory
# Display basic information and first few rows to inspect the dataset
display(df.info())
display(df.head())

# Additional Seaborn visualizations to find patterns
# Display basic information and first few rows to inspect the dataset
display(df.info())
display(df.head())
# 1. Survival rate by class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', hue='Pclass', data=df, errorbar=None, palette='coolwarm', legend=False)
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.show()

# 2. Survival rate by gender
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', hue='Sex', data=df, errorbar=None, palette='viridis', legend=False)
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Gender')
plt.show()

# 3. Fare distribution by class
plt.figure(figsize=(10, 5))
sns.boxplot(x='Pclass', y='Fare', hue='Pclass', data=df, palette='Set2', legend=False)
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.title('Fare Distribution by Passenger Class')
plt.yscale('log')  # Log scale for better visualization
plt.show()


# 3. Fare distribution by class
plt.figure(figsize=(10, 5))
sns.boxplot(x='Pclass', y='Fare', hue='Pclass', data=df, palette='Set2', legend=False)
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.title('Fare Distribution by Passenger Class')
plt.yscale('log')  # Log scale for better visualization
plt.show()


# 4. Age distribution by survival status
plt.figure(figsize=(10, 5))
sns.histplot(df, x='Age', hue='Survived', bins=30, kde=True, palette='magma', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution by Survival Status')
plt.show()


# Plot the histogram of the 'Fare' column
plt.figure(figsize=(10, 5))
sns.histplot(df['Fare'], bins=30, kde=True, color='blue')
plt.xlabel('Fare Price')
plt.ylabel('Frequency')
plt.title('Distribution of Ticket Prices (Fare)')
plt.grid(True)
plt.show()
