#%%
print("Task 1: Environment check")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("All libraries imported successfully!")
#%%
print("Task 2 & 3:  Load dataset & First look")
data = pd.read_csv('penguins.csv')
print("Dataset loaded successfully!")
print(f"First ten rows:\n{data.head(10)}")
print(f"Amounf of Rows and Colums:\n{data.shape}")
print(f"Columns Names:\n{data.columns}")
print(f"All types:\n{data.dtypes}")
# %%
print("Task 4: Missing values")
missing_counts = data.isnull().sum()
missing_percentages = (missing_counts / len(data)) * 100
print(f"Count missing values per column:\n{missing_counts}")
print(f"\nCalculate percentage of missing values:\n{missing_percentages}")
data = data.dropna()
print(f"\nShape of dataset after dropping missing values: {data.shape}")
# %%
print("Task 5: Summary statistics")
print(f"Describe Table:\n{data.describe()}")
print("\n--- Numeric Column Metrics: body_mass_g ---")
print(f"\nAverage body mass (g): {data['body_mass_g'].mean():.2f}")
print(f"Median body mass (g): {data['body_mass_g'].median():.2f}")
print(f"Standard deviation of body mass (g): {data['body_mass_g'].std():.2f}")
q1 = data['body_mass_g'].quantile(0.25)
q3 = data['body_mass_g'].quantile(0.75)
iqr = q3 - q1
print(f"Interquartile Range (IQR) body mass (g): {iqr:.2f}")
print("\n--- Categorical Column Metrics: species ---")
print(f"\nValue counts for penguin species:{data['species'].value_counts()}")
# %%
print("Task 6: Three visualizations")
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x='flipper_length_mm', kde=True, color='skyblue')
plt.title('Distribution of Penguin Flipper Lengths')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Frequency / Count')
plt.show()

print("Interpretation 1 (Histogram): The histogram shows a slightly bimodal (two-peaked) distribution, suggesting that the dataset contains distinct groups, likely corresponding to the different penguin species.")

plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x='species', y='body_mass_g', hue='species', palette='Set2', legend=False)
plt.title('Body Mass Distribution by Penguin Species')
plt.xlabel('Penguin Species')
plt.ylabel('Body Mass (g)')
plt.show()

print("Interpretation 2 (Boxplot): The boxplot clearly shows that Gentoo penguins are significantly heavier on average than both Adelie and Chinstrap penguins, which share a similar weight range.")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='bill_length_mm', y='bill_depth_mm', hue='species', palette='Set1')
plt.title('Bill Length vs. Bill Depth across Species')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.legend(title='Species')
plt.show()

print("Interpretation 3 (Scatter Plot): The scatter plot reveals distinct physical clusters for each species, such as Adelie penguins having short, deep bills compared to Gentoo penguins having long, shallow bills.")
# %%
print("Task 7: Two questions")
print("Question 1: Which penguin species has the longest average flipper length?")
flipper_by_species = data.groupby('species')['flipper_length_mm'].mean().sort_values(ascending=False)
print(f"Data Output:\n{flipper_by_species}")
print("Answer: Gentoo penguins have the longest flippers by a wide margin, averaging over 217 mm.\n")
print("-" * 80)

print("Question 2: How does the average body mass differ between male and female penguins?")
mass_by_sex = data.groupby('sex')['body_mass_g'].mean()
print(f"Data Output:\n{mass_by_sex}")
print("Answer: Across all species, male penguins are noticeably heavier on average than female penguins.")
print("-" * 80)
# %%
print("Task 8: Reflection ")
print("The trickiest part of this assignment was writing the code for the grouped charts, but I was genuinely surprised by how clearly the physical measurements separated the penguin species into distinct clusters. A logical next step would be using this clean dataset to build a machine learning model that can automatically predict a penguin's species. For this project, I used an AI assistant to brainstorm ideas for the visualizations, and I am really happy that I got to practice and write the actual code myself.")