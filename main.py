#%%

import preprocessing0
import preprocessing

path="german_credit_data.csv"
df_credit = preprocessing0.preprocess_raw(path)
# %%
print(df_credit)
# %%
X_train, X_test, y_train, y_test = preprocessing0.split_train_test(df_credit)

# %%
preprocessing.pipeline(X_train, X_test, y_train, y_test)
# %%
