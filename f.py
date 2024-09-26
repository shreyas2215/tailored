import deeplake

# Load the DeepLake dataset
ds = deeplake.load('hub://activeloop/wiki-art')

# Print the number of samples and fields in the dataset
print(f"Number of samples: {len(ds)}")
print(f"Fields in the dataset: {ds.keys()}")  # Use ds.keys() to get field names

# Check the first few entries
for i in range(min(5, len(ds))):  # Print first 5 entries
    entry = ds[i]
    print(f"Entry {i}:")
    for key in entry.keys():
        print(f"  {key}: {entry[key]}")
