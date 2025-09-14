import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use('default')

print("Loading data...")


df = pd.read_csv('sample-data-audio - 2-months.csv')
print(f"starting data shape: {df.shape}")

print("Missing values:")
print(df.isnull().sum())

df['pincode'] = df['pincode'].fillna('Unknown')
df['city'] = df['city'].fillna('Unknown')
df['state'] = df['state'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

df = df.drop_duplicates()
print(f"Data shape after cleaning: {df.shape}")

# Convert date and ensure price is numeric
df['date'] = pd.to_datetime(df['date'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Create helper columns
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['month_year'] = df['date'].dt.to_period('M')
df['quantity'] = 1  # Assuming 1 item per transaction
df['revenue'] = df['price'] * df['quantity']

print("Data cleaning completed!")

# 2. CATEGORY & BRAND DYNAMICS
print("\nAnalyzing categories and brands...")

# Category analysis
category_sales = df.groupby('level2_name')['revenue'].sum().sort_values(ascending=False)
category_counts = df.groupby('level2_name')['quantity'].sum().sort_values(ascending=False)

# Calculate cumulative percentage for Pareto analysis
category_sales_df = pd.DataFrame({
    'revenue': category_sales,
    'quantity': category_counts
})
category_sales_df['cumulative_pct'] = (category_sales_df['revenue'].cumsum() / 
                                      category_sales_df['revenue'].sum() * 100)

# Brand analysis
brand_sales = df.groupby('brand')['revenue'].sum().sort_values(ascending=False).head(10)
brand_counts = df.groupby('brand')['quantity'].sum().sort_values(ascending=False).head(10)

# Visualizations for Category & Brand
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Top 10 categories by revenue
top_categories = category_sales.head(10)
axes[0,0].bar(range(len(top_categories)), top_categories.values)
axes[0,0].set_title('Top 10 Categories by Revenue')
axes[0,0].set_xlabel('Categories')
axes[0,0].set_ylabel('Revenue (₹)')
axes[0,0].set_xticks(range(len(top_categories)))
axes[0,0].set_xticklabels(top_categories.index, rotation=45, ha='right')

# Top 10 brands by revenue
axes[0,1].bar(range(len(brand_sales)), brand_sales.values)
axes[0,1].set_title('Top 10 Brands by Revenue')
axes[0,1].set_xlabel('Brands')
axes[0,1].set_ylabel('Revenue (₹)')
axes[0,1].set_xticks(range(len(brand_sales)))
axes[0,1].set_xticklabels(brand_sales.index, rotation=45, ha='right')

# Pareto chart for categories
pareto_data = category_sales_df.head(10)
axes[1,0].bar(range(len(pareto_data)), pareto_data['revenue'])
ax_twin = axes[1,0].twinx()
ax_twin.plot(range(len(pareto_data)), pareto_data['cumulative_pct'], 
             color='red', marker='o')
axes[1,0].set_title('Pareto Analysis - Categories')
axes[1,0].set_ylabel('Revenue (₹)')
ax_twin.set_ylabel('Cumulative %')

# Category pie chart
axes[1,1].pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
axes[1,1].set_title('Revenue Distribution by Categories')

plt.tight_layout()
plt.savefig('category_brand_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. PRICE SENSITIVITY ANALYSIS
print("\nAnalyzing price sensitivity...")

# Create price bins
df['price_bin'] = pd.cut(df['price'], 
                        bins=[0, 1000, 5000, float('inf')], 
                        labels=['Low', 'Medium', 'High'])

# Analyze demand by price range
price_demand = df.groupby('price_bin').agg({
    'quantity': 'sum',
    'revenue': 'sum',
    'price': 'mean'
})

print("Price Range Analysis:")
print(price_demand)

# 4. CUSTOMER BEHAVIOR (SIMPLE RFM)
print("\nAnalyzing customer behavior...")

# Calculate RFM metrics
current_date = df['date'].max()

rfm = df.groupby('user').agg({
    'date': lambda x: (current_date - x.max()).days,  # Recency
    'user': 'count',  # Frequency
    'revenue': 'sum'  # Monetary
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Simple 3-bucket segmentation
rfm['Value_Segment'] = 'Low Value'
rfm.loc[rfm['Monetary'] >= rfm['Monetary'].quantile(0.66), 'Value_Segment'] = 'High Value'
rfm.loc[(rfm['Monetary'] >= rfm['Monetary'].quantile(0.33)) & 
         (rfm['Monetary'] < rfm['Monetary'].quantile(0.66)), 'Value_Segment'] = 'Medium Value'

print("\nCustomer Value Distribution:")
print(rfm['Value_Segment'].value_counts())
print(f"\nAverage spend by segment:")
print(rfm.groupby('Value_Segment')['Monetary'].mean().round(2))

# 5. MONTHLY TRENDS
print("\nAnalyzing monthly trends...")

monthly_sales = df.groupby('month_year').agg({
    'revenue': 'sum',
    'quantity': 'sum',
    'user': 'nunique'
})

print("Monthly Performance:")
print(monthly_sales)


# 6. KEY VISUALIZATIONS (5 Charts Only)
print("\nGenerating key visualizations...")

# Create 5 key charts
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Chart 1: Top Categories by Revenue
top_categories = category_sales.head(10)
axes[0,0].bar(range(len(top_categories)), top_categories.values)
axes[0,0].set_title('Top 10 Categories by Revenue')
axes[0,0].set_xlabel('Categories')
axes[0,0].set_ylabel('Revenue (₹)')
axes[0,0].set_xticks(range(len(top_categories)))
axes[0,0].set_xticklabels(top_categories.index, rotation=45, ha='right')

# Chart 2: Top Brands by Revenue
axes[0,1].bar(range(len(brand_sales)), brand_sales.values)
axes[0,1].set_title('Top 10 Brands by Revenue')
axes[0,1].set_xlabel('Brands')
axes[0,1].set_ylabel('Revenue (₹)')
axes[0,1].set_xticks(range(len(brand_sales)))
axes[0,1].set_xticklabels(brand_sales.index, rotation=45, ha='right')

# Chart 3: Price Range vs Quantity
axes[0,2].bar(price_demand.index, price_demand['quantity'])
axes[0,2].set_title('Quantity Sold by Price Range')
axes[0,2].set_ylabel('Total Quantity Sold')

# Chart 4: Monthly Sales Trend
axes[1,0].plot(monthly_sales.index.astype(str), monthly_sales['revenue'], marker='o')
axes[1,0].set_title('Monthly Revenue Trend')
axes[1,0].set_ylabel('Revenue (₹)')
axes[1,0].tick_params(axis='x', rotation=45)

# Chart 5: Customer Segments
segment_counts = rfm['Value_Segment'].value_counts()
axes[1,1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
axes[1,1].set_title('Customer Value Segments')

# Remove empty subplot
fig.delaxes(axes[1,2])

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Total Transactions: {len(df):,}")
print(f"Total Revenue: ₹{df['revenue'].sum():,.2f}")
print(f"Unique Customers: {df['user'].nunique():,}")
print(f"Unique Products: {df['product_id'].nunique():,}")
print(f"Unique Brands: {df['brand'].nunique():,}")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Average Price: ₹{df['price'].mean():,.2f}")
print(f"Median Price: ₹{df['price'].median():,.2f}")
print(f"Standard Deviation of Price: ₹{df['price'].std():,.2f}")
print(f"Minimum Price: ₹{df['price'].min():,.2f}")
print(f"Maximum Price: ₹{df['price'].max():,.2f}")


print("\nTop 5 Categories by Revenue:")
for i, (category, revenue) in enumerate(category_sales.head(5).items(), 1):
    print(f"{i}. {category}: ₹{revenue:,.2f}")

print("\nTop 5 Brands by Revenue:")
for i, (brand, revenue) in enumerate(brand_sales.head(5).items(), 1):
    print(f"{i}. {brand}: ₹{revenue:,.2f}")

print("\nCustomer Segments:")
for segment, count in segment_counts.items():
    print(f"{segment}: {count} customers ({count/len(rfm)*100:.1f}%)")

print("\nAnalysis completed successfully!")
