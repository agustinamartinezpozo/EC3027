### EC3027: Data Analytics in Practice ###
### Lecture 1: Bank Marketing dataset ###

# Data: UCI Bank Marketing (Moro, Cortez & Rita, 2014), CC BY 4.0
# https://archive.ics.uci.edu/dataset/222/bank+marketing
# Task: predict whether a client subscribes to a term deposit (yes/no)

# Requirement:
# pip install ucimlrepo

# %% Libraries

import os
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# Folder for exported figures
os.makedirs('figures', exist_ok=True)

# %% 1. Load the data

bank = fetch_ucirepo(id=222)
X = bank.data.features.copy()
y_raw = bank.data.targets.iloc[:, 0].copy()

print(X.head())
print(f'\nNumber of observations: {len(X)}')
print('\nOutcome distribution:')
print(y_raw.value_counts(normalize=True))

# Duration is only known after the phone call, including it would create data leakage.
X = X.drop(columns='duration')

# %% 2. Prepare the outcome variable: no = 0 and yes = 1

y = (y_raw == 'yes').astype(int).to_numpy()

# %% 3. Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

print(f'\nTraining observations: {len(X_train)}')
print(f'Test observations: {len(X_test)}')

# %% 4. Preprocessing

# Numerical vars: standardized. Categorical vars: one-hot encoding.
preprocessor = ColumnTransformer([
    ('numeric',
     StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    ('categorical',
     OneHotEncoder(handle_unknown='ignore'),
     make_column_selector(dtype_exclude=np.number))])

# %% 5. Logistic regression pipeline

model = Pipeline([('preprocessor', preprocessor),
                  ('logistic_regression', LogisticRegression(max_iter=1000))])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# %% 6. Test accuracy

test_accuracy = accuracy_score(y_test, y_pred)
print(f'\nTest accuracy: {test_accuracy:.3f}')

# %% 7. K-fold cross-validation

# The training sample is divided into 10 folds.
# The model is trained on 9 folds and validated on the remaining fold.

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

print('\nCross-validation accuracy scores:')
print(np.round(cv_scores, 3))

print(
    f'Mean cross-validation accuracy: '
    f'{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}'
)

# %% 8. Learning curve

train_sizes, train_scores, validation_scores = learning_curve(
    model,
    X_train,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

training_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)

plt.figure(figsize=(7, 5))

plt.plot(
    train_sizes,
    training_mean,
    marker='o',
    label='Training accuracy'
)

plt.plot(
    train_sizes,
    validation_mean,
    marker='s',
    label='Validation accuracy'
)

plt.xlabel('Number of training observations')
plt.ylabel('Accuracy')
plt.ylim([0.875, 0.925])
plt.title('Learning curve')
plt.legend()
plt.grid(False)
plt.tight_layout()

plt.savefig(
    'figures/lec1_learning_curve.png',
    dpi=300
)

plt.show()

# %% 9. Confusion matrix

confmat = confusion_matrix(y_test, y_pred)

print('\nConfusion matrix:')
print(confmat)

# %% 10. Evaluation metrics

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'\nAccuracy:  {test_accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall:    {recall:.3f}')
print(f'F1 score:  {f1:.3f}')

# %% 11. ROC curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f'Logistic regression (AUC = {roc_auc:.2f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Random prediction (AUC = 0.50)'
)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()

plt.savefig(
    'figures/lec1_roc_curve.png',
    dpi=300
)

plt.show()

