import pandas as pd
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset('titanic')

# -----------------------------
# 🔍 1. REMOVE DUPLICATES
# -----------------------------
print(f"Before removing duplicates: {df.shape}")
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# -----------------------------
# 🧹 2. HANDLE MISSING VALUES
# -----------------------------

# Check null values
print("\nMissing values before:")
print(df.isnull().sum())

# Fill 'age' with median
df['age'] = df['age'].fillna(df['age'].median())

# Fill 'embarked' with mode (most frequent value)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Fill 'deck' with 'Unknown' (or you could drop it if too sparse)
df['deck'] = df['deck'].fillna('Unknown')

# Drop 'embark_town' since it's similar to 'embarked' and has missing values
df = df.drop(columns='embark_town')

# Drop any remaining rows with missing values (if any)
df = df.dropna()

# Check again
print("\nMissing values after:")
print(df.isnull().sum())

# -----------------------------
# 🧾 3. FIX INCONSISTENT FORMATS
# -----------------------------

# Convert 'sex' and 'embarked' to lowercase for consistency
df['sex'] = df['sex'].str.lower()
df['embarked'] = df['embarked'].str.lower()

# Convert categorical to category dtype
df['sex'] = df['sex'].astype('category')
df['embarked'] = df['embarked'].astype('category')
df['class'] = df['class'].astype('category')
df['who'] = df['who'].astype('category')

# Convert 'fare' to 2 decimal float (optional, for formatting)
df['fare'] = df['fare'].round(2)

# Preview cleaned data
print("\nCleaned data preview:")
print(df.head())
