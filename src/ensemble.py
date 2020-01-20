"""
Contains the final model used to identify kermit. This is an ensemble of the best model from `audio_model_test.py` & the visual model.
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
from sklearn.metrics import recall_score, accuracy_score, f1_score
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
def get_audio_model():
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

    return model

# %% Create the visual model
def get_visual_model():
    raise NotImplementedError


# %% Ensemble
class FullModel:
    def __init__(self):
        self.audio_model_ = get_audio_model()
        self.visual_model_ = get_visual_model()

    def fit(self, x, y):
        self.audio_model_.fit(x, y)
        self.visual_model_.fit(x, y)

    def predict_proba(self, x):
        proba_audio = self.audio_model_.predict_proba(x)[:, 0]
        proba_vis = self.visual_model_.predict_proba(x)[:, 0]

        a = 0.1
        proba = proba_audio * a + proba_vis * (1-a)

        return proba

    def predict(self, x):
        proba = self.predict_proba(x)
        return proba < 0.5


# Evaluate
model = FullModel()

model.fit(train_x, train_y)
pred_y = model.predict(test_x)


print(f'Accuracy: {accuracy_score(test_y, pred_y)}')
print(f'Recall:   {recall_score(test_y, pred_y)}')
print(f'F1:       {f1_score(test_y, pred_y)}')



# %%
