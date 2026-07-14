### EC3027: Data Analytics in Practice ###
### Lecture 2: ACSIncome (US Census) ###

# Data: ACSIncome from the folktables package (Ding et al., 2021),
# https://github.com/socialfoundations/folktables
# Task: predict whether a person's income exceeds $50,000.

# Requirement: 
# pip install folktables

# Libraries and Packages
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

os.makedirs('figures', exist_ok=True)

# %% Loading the ACSIncome data (Rhode Island, 2018)

from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year='2018',
                            horizon='1-Year',
                            survey='person')
acs_data = data_source.get_data(states=['RI'], download=True)

X_df, y_df, _ = ACSIncome.df_to_pandas(acs_data)
y_all = y_df.iloc[:, 0].astype(int).values   # 1 if income > $50,000

print(X_df.head())
print(X_df.shape)
print(f'Share with income > $50k: {y_all.mean():.3f}')

# %% A two-feature version for visualization

# For the decision-region plots we keep two interpretable features: age and weekly hours of work.
X2 = X_df[['AGEP', 'WKHP']].values
y = y_all

X_train, X_test, y_train, y_test = train_test_split(X2, y,
                                                    test_size=0.30,
                                                    stratify=y,
                                                    random_state=1)

# Standardize 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train) # fit_transform for train
X_test_std = sc.transform(X_test)      # transform (not fit) for test

# Scatter plot of the raw data
plt.figure(figsize=(7, 5))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            c='blue', marker='^', alpha=0.3, s=12, label='<= $50k')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='green', marker='o', alpha=0.3, s=12, label='> $50k')
plt.xlabel('Age [years]')
plt.ylabel('Usual weekly working hours')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/lec2_data.png', dpi=300)
plt.show()

# %% Helper: plotting decision regions (adapted from ML-PSL, ch. 3)

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    markers = ('^', 'o', 's', 'v', '<')
    colors = ('blue', 'green', 'red', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.4,
                    c=colors[idx],
                    marker=markers[idx],
                    s=20,
                    label=f'Class {cl}')

    if test_idx is not None:
        X_test_pts = X[test_idx, :]
        plt.scatter(X_test_pts[:, 0], X_test_pts[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=80, label='Test set')

# To keep the plots readable we draw a random subsample of the points.
rng = np.random.RandomState(1)
sub = rng.choice(len(X_train_std), size=min(800, len(X_train_std)),
                 replace=False)
sub_test = rng.choice(len(X_test_std), size=min(200, len(X_test_std)),
                      replace=False)
X_plot = np.vstack((X_train_std[sub], X_test_std[sub_test]))
y_plot = np.hstack((y_train[sub], y_test[sub_test]))
test_range = range(len(sub), len(sub) + len(sub_test))

# %% Logistic Regression

lr = LogisticRegression(C=100.0, max_iter=1000)
lr.fit(X_train_std, y_train)

print(f'Logit train accuracy: {lr.score(X_train_std, y_train):.3f}')
print(f'Logit test accuracy:  {lr.score(X_test_std, y_test):.3f}')

plot_decision_regions(X_plot, y_plot, classifier=lr,
                      test_idx=test_range)
plt.xlabel('Age [standardized]')
plt.ylabel('Weekly hours [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/lec2_logit_regions.png', dpi=300)
plt.show()


# %% Single-layer NN: the perceptron 

# Remember, if classes are NOT linearly separable, it never converges exactly!
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# Compare its accuracy with the logistic regression above:
print(f'Perceptron test accuracy: {ppn.score(X_test_std, y_test):.3f}')
print(f'Logit test accuracy:      {lr.score(X_test_std, y_test):.3f}')

plot_decision_regions(X_plot, y_plot, classifier=ppn,
                      test_idx=test_range)
plt.xlabel('Age [standardized]')
plt.ylabel('Weekly hours [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/lec2_perceptron_regions.png', dpi=300)
plt.show()

# %% Single-layer NN: Adaline (Adaptive Linear Neuron)

# scikit-learn has no class literally called "Adaline": it is the textbook
# name for a linear neuron trained by gradient descent to minimize the
# sum of squared errors between the linear activation phi(z) = z and the
# true label (the Widrow-Hoff / LMS rule), with the 0/1 step applied only
# at prediction time. SGDClassifier(loss='squared_error') implements
# exactly that rule, so we reuse it here instead of coding our own
# gradient-descent loop.
ada = SGDClassifier(loss='squared_error', learning_rate='constant',
                    eta0=0.01, random_state=1)
ada.fit(X_train_std, y_train)

print(f'Adaline test accuracy:    {ada.score(X_test_std, y_test):.3f}')
print(f'Perceptron test accuracy: {ppn.score(X_test_std, y_test):.3f}')
print(f'Logit test accuracy:      {lr.score(X_test_std, y_test):.3f}')

plot_decision_regions(X_plot, y_plot, classifier=ada,
                      test_idx=test_range)
plt.xlabel('Age [standardized]')
plt.ylabel('Weekly hours [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/lec2_adaline_regions.png', dpi=300)
plt.show()

# %% Support Vector Machine

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=1)
svm.fit(X_train_std, y_train)

print(f'SVM train accuracy: {svm.score(X_train_std, y_train):.3f}')
print(f'SVM test accuracy:  {svm.score(X_test_std, y_test):.3f}')

plot_decision_regions(X_plot, y_plot, classifier=svm,
                      test_idx=test_range)
plt.xlabel('Age [standardized]')
plt.ylabel('Weekly hours [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/lec2_svm_regions.png', dpi=300)
plt.show()

# %% K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)

print(f'KNN train accuracy: {knn.score(X_train_std, y_train):.3f}')
print(f'KNN test accuracy:  {knn.score(X_test_std, y_test):.3f}')

plot_decision_regions(X_plot, y_plot, classifier=knn,
                      test_idx=test_range)
plt.xlabel('Age [standardized]')
plt.ylabel('Weekly hours [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/lec2_knn_regions.png', dpi=300)
plt.show()



# %% Using all features (with proper encoding of the categorical codes)

# In the two-feature model we ignored education, sex, occupation, etc.
# Here we use a richer set. SCHL is ordered (educational attainment),
# so we keep it numeric; the nominal codes are one-hot encoded.
X_full = X_df[['AGEP', 'WKHP', 'SCHL', 'COW', 'MAR', 'SEX']]
cat_cols = ['COW', 'MAR', 'SEX']

# A few records can have missing codes; drop them (and align y)
keep = X_full.notna().all(axis=1)
X_full, y_full = X_full[keep], y_all[keep.values]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.30, stratify=y_full, random_state=1)

pre = ColumnTransformer([
    ('num', StandardScaler(), ['AGEP', 'WKHP', 'SCHL']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

for name, est in [('Logistic regression',
                   LogisticRegression(max_iter=1000)),
                  ('SVM (rbf)', SVC(kernel='rbf')),
                  ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5))]:
    pipe = Pipeline([('pre', pre), ('clf', est)])
    pipe.fit(X_train_f, y_train_f)
    print(f'{name}: test accuracy = '
          f'{pipe.score(X_test_f, y_test_f):.3f}')
