import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the behavioral dataset
behavior_data = pd.read_csv('../dataset/cow_behavior_dataset.csv')

# Convert categorical columns into numerical
behavior_features = behavior_data.drop('Label', axis=1)
behavior_labels = behavior_data['Label'].map({'healthy': 0, 'infected': 1})  # Convert labels to 0 and 1

# One-Hot Encode categorical variables
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
behavior_features_encoded = encoder.fit_transform(behavior_features)

# Display the shape of the encoded features
print(f"Encoded feature shape: {behavior_features_encoded.shape}")

# Split behavioral data into train and validation sets
X_train_behavior, X_val_behavior, y_train_behavior, y_val_behavior = train_test_split(
    behavior_features_encoded, behavior_labels, test_size=0.2, random_state=42
)

# Standardize the data (Optional)
scaler = StandardScaler()
X_train_behavior = scaler.fit_transform(X_train_behavior)
X_val_behavior = scaler.transform(X_val_behavior)

# Build the behavioral model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train_behavior.shape[1],)))  # Updated to match the shape
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary output

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_behavior,
    y_train_behavior,
    validation_data=(X_val_behavior, y_val_behavior),
    epochs=20,
    batch_size=32
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(X_val_behavior, y_val_behavior)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Save the behavioral model
model.save('behavioral_lsd_model.h5')
print("Behavioral model has been trained and saved as 'behavioral_lsd_model.h5'.")
