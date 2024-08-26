# %%

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

import optuna
import lightgbm as lgb

from matplotlib import pyplot as plt

# %%

# Load data
sample = pd.read_parquet('/groups/hep/kinch/H_Zg/samples_processed/HZeeg_ggF_MC_reduced_Zmodel_eepairs_23August.parquet') 

isZ = sample[sample['isZ'] == 1]
isNotZ = sample[sample['isZ'] == 0]
isNotZ_sampled = isNotZ.sample(frac=0.01, random_state=42) 
small_sample = pd.concat([isZ, isNotZ_sampled])
print(f'full sample size:{len(sample)}, number of Z:{len(isZ)}, number of not Z:{len(isNotZ)}')
print(f'small sample size:{len(small_sample)}, number of Z:{len(isZ)}, number of not Z:{len(isNotZ_sampled)}')

# %%
r = np.random
r.seed(42)

input_data = small_sample.drop(columns=['isZ','m_ee','event_index','el_index','el1_truthPdgId','el1_truthType','el1_truthOrigin','el2_truthPdgId','el2_truthType','el2_truthOrigin','el1_phi','el2_phi'])
truth_data= small_sample['isZ']

X_train, X_test, y_train, y_test = train_test_split(input_data, truth_data, test_size=0.25, random_state=42)

# %%
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 0, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'drop_rate': trial.suggest_uniform('drop_rate', 0.005, 0.4),
        'verbose': 0
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    return auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# %%
print(study.best_params)
print(study.best_value)
print(study.best_trial.number)



# %%
values = [trial.value for trial in study.trials]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(values)
ax.set_xlabel('Trial')
ax.set_ylabel('AUC')
ax.set_title('Optimization Metric Across Trials')

fig.tight_layout()
fig.savefig('/groups/hep/kinch/thesis/code/plots/optuna.png')

# %%

model = lgb.LGBMClassifier(**study.best_params)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# %%
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2e})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig('/groups/hep/kinch/thesis/code/plots/roc_curve.png')

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_proba[y_test == 0], bins=100, alpha=0.5, label='False')
ax.hist(y_pred_proba[y_test == 1], bins=100, alpha=0.5, label='True')
ax.legend()
ax.set_xlabel('Prediction Probability')
fig.tight_layout()
fig.savefig('/groups/hep/kinch/thesis/code/plots/prediction_probability.png')

# %%
# fit model to full data set

full_sample = pd.read_parquet('/groups/hep/kinch/H_Zg/samples_processed/HZeeg_ggF_MC_reduced_Zmodel_eepairs_23August.parquet') 

full_labels = full_sample['isZ']
full_input = full_sample.drop(columns=['isZ','m_ee','event_index','el_index','el1_truthPdgId','el1_truthType','el1_truthOrigin','el2_truthPdgId','el2_truthType','el2_truthOrigin','el1_phi','el2_phi'])

full_model = lgb.LGBMClassifier(**study.best_params)
full_model.fit(full_input, full_labels)

y_pred_full = full_model.predict(full_input)
y_pred_small_full = model.predict(full_input)

y_pred_proba_full = full_model.predict_proba(full_input)[:,1]
y_pred_proba_small_full = model.predict_proba(full_input)[:,1]

# %%
# roc curve and histogram for full data set:

fpr_full, tpr_full, _ = roc_curve(full_labels, y_pred_proba_full)
roc_auc_full = auc(fpr_full, tpr_full)

fpr_small_full, tpr_small_full, _ = roc_curve(full_labels, y_pred_proba_small_full)
roc_auc_small_full = auc(fpr_small_full, tpr_small_full)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr_full, tpr_full, label=f'ROC curve (area = {roc_auc_full:.2e})')
ax.plot(fpr_small_full, tpr_small_full, label=f'ROC curve small model (area = {roc_auc_small_full:.2e})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig('/groups/hep/kinch/thesis/code/plots/roc_curve_full.png')


fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_proba_full[full_labels == 0], bins=100, alpha=0.5, label='False')
ax.hist(y_pred_proba_full[full_labels == 1], bins=100, alpha=0.5, label='True')

ax.hist(y_pred_proba_small_full[full_labels == 0], bins=100, alpha=0.5, label='False small model')
ax.hist(y_pred_proba_small_full[full_labels == 1], bins=100, alpha=0.5, label='True small model')
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('Prediction Probability')
fig.tight_layout()
fig.savefig('/groups/hep/kinch/thesis/code/plots/prediction_probability_full.png')



# %%
# plot feature importance
print(sample.columns)
fig, ax = plt.subplots(figsize=(10, 6))
lgb.plot_importance(model, ax=ax)
fig.tight_layout()
fig.savefig('/groups/hep/kinch/thesis/code/plots/feature_importance.png')

fig, ax = plt.subplots(figsize=(10, 6))
lgb.plot_importance(full_model, ax=ax)
fig.tight_layout()
fig.savefig('/groups/hep/kinch/thesis/code/plots/feature_importance_full.png')