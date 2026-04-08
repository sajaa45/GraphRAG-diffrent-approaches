import pandas as pd
from collections import Counter

# File paths
sub_file = r"C:\Users\LENOVO\Downloads\sub.txt"
num_file = r"C:\Users\LENOVO\Downloads\num.txt"

# Step 1: Read sub.txt and filter by sic = 6798
print("Reading sub.txt...")
sub_df = pd.read_csv(sub_file, sep='\t', low_memory=False)  # Adjust separator if needed

# Filter for sic = 6798
filtered_adsh = sub_df[sub_df['sic'] == 6798]['adsh'].unique()
print(f"Found {len(filtered_adsh)} unique adsh values with sic=6798")

# Step 2: Read num.txt
print("\nReading num.txt...")
num_df = pd.read_csv(num_file, sep='\t', low_memory=False)

# Filter num.txt for matching adsh values
num_filtered = num_df[num_df['adsh'].isin(filtered_adsh)]
print(f"Found {len(num_filtered)} rows in num.txt for these adsh values")

# Step 3: Find tags that exist for ALL adsh values
print("\nFinding tags that exist for all adsh values...")
adsh_tag_counts = num_filtered.groupby('tag')['adsh'].nunique()
tags_for_all = adsh_tag_counts[adsh_tag_counts == len(filtered_adsh)].index.tolist()

print(f"Found {len(tags_for_all)} tags that appear for all {len(filtered_adsh)} adsh values")

# Step 4: Filter num.txt to only these tags
final_result = num_filtered[num_filtered['tag'].isin(tags_for_all)]

# Display results
print(f"\nFinal filtered dataset has {len(final_result)} rows")
print("\nFirst few rows:")
print(final_result.head())

# Optional: Save to file
output_file = r"C:\Users\LENOVO\Downloads\filtered_num.txt"
final_result.to_csv(output_file, sep='\t', index=False)
print(f"\nSaved filtered results to {output_file}")

# Show some statistics
print("\nTag statistics:")
for tag in tags_for_all[:10]:  # Show first 10 tags
    count = final_result[final_result['tag'] == tag]['adsh'].nunique()
    print(f"  {tag}: appears for {count} adsh values")