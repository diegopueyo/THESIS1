# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load the dataset (you already did this!)
file_path = r"C:\Users\diego\OneDrive\Escritorio\TESIS\datasets\UserBehavior.csv"

# Define column names manually because the file has no header
column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

# Read the dataset
df = pd.read_csv(file_path, names=column_names)
# 3. Check basic information
print(df.info())
# 4. Show the first few rows
print(df.head())

# What are the different types of behaviors?
print(df['behavior_type'].unique())
# How many times each behavior happens?
print(df['behavior_type'].value_counts())


#PLOT 1 the counts of each behavior
import matplotlib.pyplot as plt
import seaborn as sns

# Set a prettier style
sns.set(style="whitegrid")

# Plot
plt.figure(figsize=(8,6))
ax = df['behavior_type'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of User Behaviors', fontsize=16)
plt.xlabel('Behavior Type', fontsize=14)
plt.ylabel('Number of Events', fontsize=14)
plt.xticks(rotation=0)

# Optional: Add exact numbers on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()



# How many unique users?
#If the number of items or users is small → there is more risk of echo chambers.
#If there are many categories and users only explore a few → it can cause filter bubbles.

num_users = df['user_id'].nunique()
print(f"Number of unique users: {num_users:,}")

# How many unique items?
num_items = df['item_id'].nunique()
print(f"Number of unique items: {num_items:,}")

# How many unique categories?
num_categories = df['category_id'].nunique()
print(f"Number of unique categories: {num_categories:,}")


# Find most viewed items
most_viewed_items = df[df['behavior_type'] == 'pv']['item_id'].value_counts().head(10)
print("Top 10 most viewed items:")
print(most_viewed_items)

# Find most purchased items
most_purchased_items = df[df['behavior_type'] == 'buy']['item_id'].value_counts().head(10)
print("Top 10 most purchased items:")
print(most_purchased_items)


# View item popularity distribution
item_popularity = df[df['behavior_type'] == 'pv']['item_id'].value_counts()

# Plot
plt.figure(figsize=(10,6))
plt.hist(item_popularity, bins=100, log=True)
plt.title('Distribution of Product Popularity (Log Scale)')
plt.xlabel('Number of Views per Item')
plt.ylabel('Number of Items (Log Scale)')
plt.show()



# Find the most viewed categories
most_viewed_categories = df[df['behavior_type'] == 'pv']['category_id'].value_counts().head(10)
print("Top 10 most viewed categories:")
print(most_viewed_categories)


# Plot the top viewed categories
most_viewed_categories.plot(kind='bar', figsize=(10,6), color='lightgreen')
plt.title('Top 10 Most Viewed Categories')
plt.xlabel('Category ID')
plt.ylabel('Number of Views')
plt.xticks(rotation=45)
plt.show()



#BEHAVIOUR ANALYSIS
# How many different items does each user view?
user_item_counts = df[df['behavior_type'] == 'pv'].groupby('user_id')['item_id'].nunique()

# Basic statistics
print("Statistics on number of different items viewed per user:")
print(user_item_counts.describe())

# How many different categories does each user view?
user_category_counts = df[df['behavior_type'] == 'pv'].groupby('user_id')['category_id'].nunique()

# Basic statistics
print("Statistics on number of different categories viewed per user:")
print(user_category_counts.describe())


plt.figure(figsize=(10,6))
plt.hist(user_item_counts, bins=100, log=True, color='skyblue')
plt.title('Distribution of Number of Different Items Viewed per User (Log Scale)')
plt.xlabel('Number of Different Items')
plt.ylabel('Number of Users (Log Scale)')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(user_category_counts, bins=50, log=True, color='lightgreen')
plt.title('Distribution of Number of Different Categories Viewed per User (Log Scale)')
plt.xlabel('Number of Different Categories')
plt.ylabel('Number of Users (Log Scale)')
plt.show()





# 1. Convert timestamp
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['date'] = df['datetime'].dt.date

# 2. Filter a narrower date range
df_filtered = df[(df['datetime'] >= '2017-11-21') & (df['datetime'] <= '2017-12-03')]

# 3. Group by date and behavior
daily_behavior_filtered = df_filtered.groupby(['date', 'behavior_type']).size().unstack(fill_value=0)

# 4. Plot
plt.figure(figsize=(14,8))
daily_behavior_filtered.plot()
plt.title('Daily Evolution of User Behaviors (Focused Dates)')
plt.xlabel('Date')
plt.ylabel('Number of Events')
plt.legend(title='Behavior Type')
plt.grid(True)

# 🔥 Rotate x-axis labels
plt.xticks(rotation=45)

plt.show()