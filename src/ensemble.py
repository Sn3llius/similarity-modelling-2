"""
Contains the final model used to identify kermit. This is an ensemble of the
best model from `audio_model_test.py` & the visual model.

The results from the visual model are loaded from disk. Audio predictions are
made in this file.
"""

# %% Imports
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier



# %% Load the data
root_path = Path().resolve().parent
data_path = root_path / 'data'

# Loading
def load_data():
    data = pickle.load(
        (data_path / 'audio' / 'audio_features.pickle').open('rb'))

    full_x = data['mfcc']
    full_y = data['label']

    return full_x, full_y


full_x, full_y = load_data()

# Balance the classes
# balanced_x, balanced_y = full_x, full_y
sampler = RandomUnderSampler(
    random_state = 0,
)
balanced_x, balanced_y = sampler.fit_resample(full_x, full_y)

# Train/Test split
train_x, test_x, train_y, test_y = train_test_split(
    balanced_x, balanced_y,
    test_size = 0.2,
    stratify = balanced_y,
    random_state=0,
)

# %% Create the audio model
def make_audio_model():
    model = Pipeline(
        [
            [
                'scaler',
                StandardScaler()
            ],
            [
                'classifier',
                SVC(C=1, kernel='rbf', probability=True)
            ]
        ]
    )

    model.fit(train_x, train_y)
    return model


# %% Ensemble
def ensemble_proba(preds_a, preds_b, fac):
    return preds_a * (1-fac) + preds_b * fac

# Evaluate
gt_y = np.load(data_path / 'predictions_nn' / 'y_true.npy')[:-8]

preds_visual = np.load(data_path / 'predictions_nn' / 'y_pred.npy')[:-8]

audio_model = make_audio_model()
preds_audio = audio_model.predict_proba(full_x)[:, 0]

assert preds_visual.shape == preds_audio.shape, (preds_visual.shape, preds_audio.shape)

pred_y_proba = ensemble_proba(preds_visual, preds_audio, 0.1)
pred_y = pred_y_proba > 0.5

print(f'Accuracy: {accuracy_score(gt_y, pred_y)}')
print(f'Recall:   {recall_score(gt_y, pred_y)}')
print(f'F1:       {f1_score(gt_y, pred_y)}')


# %%
import matplotlib.pyplot as plt

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

fper, tper, thresholds = roc_curve(gt_y, pred_y_proba)
plot_roc_curve(fper, tper)


# %%
