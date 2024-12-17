import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import keras_tuner as kt

def setup_logger(log_dir='./logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('GeoWindNet')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, 'geowindnet.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger

def data_loader(data_path, target_column):
    logger = setup_logger()
    try:
        logger.info("Loading dataset...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape(-1, 256, 256, 12)
        y = tf.keras.utils.to_categorical(y, 2)
        logger.info("Data loaded and preprocessed successfully.")
        return X_scaled, y
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def build_model(input_shape, dropout_rate=0.4, num_classes=2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_hypermodel(hp):
    model = models.Sequential([
        layers.Input(shape=(256, 256, 12)),
        layers.Conv2D(hp.Int('conv_units', 32, 64, step=16), (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'),
        layers.Dropout(rate=hp.Float('dropout', 0.2, 0.5, step=0.1)),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling="LOG")),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs=50):
    lr_scheduler = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.9**epoch)
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        callbacks=[lr_scheduler])
    return history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    auc_roc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def main():
    logger = setup_logger()
    try:
        data_path = './data/seafloor_data.csv'
        target_column = 'suitability'
        X, y = data_loader(data_path, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tuner = kt.RandomSearch(
            build_hypermodel,
            objective='val_accuracy',
            max_trials=5,
            directory='hyperparam_logs',
            project_name='GeoWindNet_Tuning'
        )
        tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
        best_hp = tuner.get_best_hyperparameters()[0]
        model = build_model(input_shape=(256, 256, 12), dropout_rate=best_hp['dropout'])
        model.compile(optimizer=optimizers.Adam(best_hp['learning_rate']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = train_model(model, X_train, y_train, X_test, y_test, batch_size=16)
        evaluate_model(model, X_test, y_test)
        model.save('geowindnet_final.h5')
        logger.info("Model training and evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
