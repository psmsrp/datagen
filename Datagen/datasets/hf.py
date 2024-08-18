from datasets import load_dataset

# List of dataset names and corresponding configurations
datasets = [
    # ('Salesforce/dialogstudio', 'DialogSum'),
    # ('Salesforce/dialogstudio', 'SAMSum'),
    # ('Salesforce/dialogstudio', 'ConvoSumm'),
    # ('Salesforce/dialogstudio', 'TweetSumm'),
    ('Salesforce/dialogstudio', 'chitchat-dataset'),
]

# Loop through each dataset, load it, and save it as a CSV file
for dataset_name, config_name in datasets:
    # Load the dataset with the specified configuration
    dataset = load_dataset(dataset_name, config_name)
    
    # Save each split (train, test, validation) as a separate CSV file
    for split in dataset.keys():
        # Construct the filename based on the dataset and split name
        filename = f"{config_name}_{split}.csv"
        
        # Convert the dataset split to a Pandas DataFrame
        df = dataset[split].to_pandas()
        
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        
        print(f"Saved {filename}")
