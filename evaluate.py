import time
import turicreate as tc

# Load the data
data = tc.SFrame('HongKong-dollar.sframe')

# Make a train-test split
train_data, test_data = data.random_split(0.9)

# Set the batch size and number of epochs
batch_size = 100
num_epochs = 100

# Create the model
model = tc.image_classifier.create(train_data, target='label')

# Train the model with early stopping
start_time = time.time()
for i in range(num_epochs):
    # Train for one epoch
    # model.train(train_data, max_iterations=1, batch_size=batch_size)

    # Print the number of images processed
    num_images_processed = (i + 1) * len(train_data)
    print(f'Images processed: {num_images_processed}')

# Evaluate the model on the test data
metrics = model.evaluate(test_data)

# Print the accuracy
print(f'Accuracy = {metrics["accuracy"]}')

# Calculate the training time
end_time = time.time()
training_time = end_time - start_time

# Print the training time
print(f'Training completed in {training_time:.0f} seconds')