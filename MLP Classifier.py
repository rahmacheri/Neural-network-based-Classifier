

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from scipy.io import loadmat, savemat
from sklearn.metrics import classification_report, coverage_error, label_ranking_average_precision_score, hamming_loss
import os
import time

def main(scenario_name, snr_value, top_k=44):
    # Load and reshape data
    train_X = loadmat(f'/path/{scenario_name}-{snr_value}-train/data.mat')['datat']
    train_Y = loadmat(f'/path/{scenario_name}-{snr_value}-train-labels/labels.mat')['Labels']
    valid_X = loadmat(f'/path/{scenario_name}-{snr_value}-valid/data.mat')['datat']
    valid_Y = loadmat(f'/path/{scenario_name}-{snr_value}-valid-labels/labels.mat')['Labels']
    test_X = loadmat(f'/path/{scenario_name}-{snr_value}-test/data.mat')['datat']
    test_Y = loadmat(f'/path/{scenario_name}-{snr_value}-test-labels/labels.mat')['Labels']

    train_X = np.transpose(train_X, (3, 0, 1, 2)).reshape(6000, -1)
    valid_X = np.transpose(valid_X, (3, 0, 1, 2)).reshape(2000, -1)
    test_X = np.transpose(test_X, (3, 0, 1, 2)).reshape(2000, -1)

    train_X = np.concatenate((train_X, valid_X))
    train_Y = np.concatenate((train_Y, valid_Y))

    # Define the MLP model with dropout and L2 regularization
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(train_X.shape[1],), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.8))
    model.add(Dense(train_Y.shape[1], activation='sigmoid', kernel_regularizer=l2(0.01)))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy')

    # Train the model
    model.fit(train_X, train_Y, epochs=200, batch_size=500, validation_split=0.2, verbose=1)

    print("Starting inference...")
    start_inference_time = time.time()

    # Predict labels for the test set
    y_pred_prob = model.predict(test_X)
    sorted_indices = np.argsort(-y_pred_prob, axis=1)
    y_pred = np.zeros_like(test_Y)

    for i in range(len(test_Y)):
        top_indices = sorted_indices[i, :top_k]
        y_pred[i, top_indices] = 1

    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time

    print(f"Inference Time: {inference_time} seconds", flush=True)

    # Generate a classification report
    report = classification_report(test_Y, y_pred)
    print("Classification Report:\n", report)
    coverage = coverage_error(test_Y, y_pred_prob)
    print(f"Coverage Error: {coverage:.4f}")
    avg_precision = label_ranking_average_precision_score(test_Y, y_pred_prob)
    print(f"Label Ranking Average Precision Score: {avg_precision:.4f}")
    hamming = hamming_loss(test_Y, y_pred)
    print(f"Hamming Loss: {hamming:.4f}")

    directory = "/your-path/"
    os.makedirs(directory, exist_ok=True)
    filename = f"{scenario_name}_{snr_value}_MLP.mat"
    filepath = os.path.join(directory, filename)

    try:
        savemat(filepath, {"y_pred": y_pred})
    except Exception as e:
        print("Error saving file:", e)

if __name__ == "__main__":
    scenario_name = "20"
    snr_value = "snr10"
    main(scenario_name, snr_value, top_k=44)
