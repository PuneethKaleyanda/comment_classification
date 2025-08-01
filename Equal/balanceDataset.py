import pandas as pd
from sklearn.utils import resample

# Load the dataset
train_data = pd.read_csv(r'comment_classification\train.csv')  

# Function to balance labels
def balance_labels(df, label):
    # Separate minority and majority classes
    df_majority = df[df[label] == 0]
    df_minority = df[df[label] == 1]

    # Upsample minority class to match the majority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # match majority class
                                     random_state=42)

    # Combine majority class with upsampled minority class
    return pd.concat([df_majority, df_minority_upsampled])

# Apply the function to balance each label
balanced_data = train_data.copy()
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    balanced_data = balance_labels(balanced_data, label)

# Save the balanced dataset
balanced_data.to_csv('balanced_train.csv', index=False)
print("Balanced dataset saved as 'balanced_train.csv'")
