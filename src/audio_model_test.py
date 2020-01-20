"""
Testing ground for different models. The final model is in `audio_model.py`.
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
root_path = Path('/home/jakob/Local/similarity-modelling-2')
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

# %%

def build_dnn_cb(activation='relu'):
    model = Sequential()

    model.add(Dense(15, activation=activation, input_dim=train_x.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation=activation))


    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

dnn = KerasClassifier(build_dnn_cb)


# %%
pipe = Pipeline(
    [
        [
            'scaler',
            # StandardScaler()
            MinMaxScaler()
        ],
        [
            'classifier',
            # SVC()
            RandomForestClassifier()
            # KNeighborsClassifier()
            # KerasClassifier(build_dnn_cb, epochs=10)
        ]
    ]
)


# %% Grid Search
param_grid = {
    # 'classifier__C': [0.8, 1, 1.5, 4, 10],
    # 'classifier__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),

    # 'classifier__n_estimators': (50, 100, 400, 1000),
    'classifier__criterion': ('gini', 'entropy'),
    # 'classifier__max_depth': (2, 5, 8, None),
    # 'classifier__bootstrap': (True, False),

    # 'classifier__n_neighbors': (1, 2, 3, 5, 9),
}

optimizer = GridSearchCV(
    pipe,
    param_grid,
    scoring='accuracy',
    cv=3,
    # verbose=2,
    n_jobs=1,
)

optimizer.fit(train_x, train_y)
pipe = optimizer.best_estimator_
pred_y = pipe.predict(test_x)

print(f'Accuracy: {accuracy_score(test_y, pred_y)}')
print(f'Recall:   {recall_score(test_y, pred_y)}')
print(f'F1:       {f1_score(test_y, pred_y)}')


# %%
