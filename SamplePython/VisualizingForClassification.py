import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

auto_prices = pd.read_csv('Automobile price data _Raw_.csv')
auto_prices.head(20)

credit = pd.read_csv('German_Credit.csv', header=None)
credit.head()



def plot_box(credit, cols, col_x = 'bad_credit'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col_x, col, data=credit)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
num_cols = ['loan_duration', 'loan_amount', 'payment_pcnt_income',
            'age_yrs', 'number_loans', 'depents']
plot_box(credit, num_cols)

def plot_violin(credit, cols, col_x = 'bad_credit'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col_x, col, data=credit)
        plt.xlabel(col_x)
        plt.ylabel(cols)
        plt.show()
plot_violin(credit, num_cols)




import numpy as np
cat_colors = ['checking_account_status', 'credit_history', 'purpose', 'saving_account_balance',
                'time_employed_yrs', 'gender_status', 'other_signatures', 'property',
                'other_credit_outstanding', 'home_ownership', 'job_category', 'telephone',
                'foreign_worker']
# The following block is comment as credit is not defined
credit['dummy'] = np.ones(shape = credit.shape[0])
for col in cat_colors:
    print(col)
    counts = credit[['dummy', 'bad_credit', col]].groupby(['bad_credit', col], as_index=False).count()
    temp = counts[counts['bad_credit'] == 0][[col, 'dummy']]
    _ = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    temp = counts[counts['bad_credit'] == 0][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counter fir ' + col + '\n Bad credit')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    temp = counts[counts['bad_credit'] == 1][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Count for ' + col + '\n Good credit')
    plt.show()


# Look into frequency
def count_unique(auto_prices, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(auto_prices[col].value_counts())
cat_cols = ['make', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style',
            'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders',
            'fuel_system']
count_unique(auto_prices, cat_cols)

# imbalance the label
credit_counts = credit['bad_credit'].value_counts()
print(credit_counts)
