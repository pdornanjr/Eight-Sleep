import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from data_pull import pull_data

orig_df=pull_data()


# Later Step: Move to user input/flask workspace
month_days=30
outlier_cutoff=50000


def transform_data(orig_df):
    orig_df['date'] = pd.to_datetime(orig_df['date'])
    orig_df['cac'] = orig_df['spend']/orig_df['transactions']
    factors = ['transactions', 'spend', 'visits','cac']
    grouped = orig_df.groupby(['date', 'campaign_name'])[factors].sum().reset_index()
    combined = pd.concat([grouped], ignore_index=True)
    combined = combined.sort_values(by=['campaign_name', 'date']).reset_index(drop=True)

    for factor in factors:
        combined[f'previous_day_{factor}_campaign'] = combined.groupby('campaign_name')[factor].shift(1)
        combined[f'previous_week_{factor}_campaign'] = combined.groupby('campaign_name')[factor].rolling(7, min_periods=1).sum().shift(1).reset_index(drop=True)
        combined[f'previous_month_{factor}_campaign'] = combined.groupby('campaign_name')[factor].rolling(month_days, min_periods=1).sum().shift(1).reset_index(drop=True)

    combined = combined.fillna(0)

    return combined


def generate_transactions_vs_spend( max_spend=outlier_cutoff, increment=100):
    transactions_model = model['transactions']
    visits_model = model['visits']
    spend_values = list(range(0, max_spend + increment, increment))

    transactions_predictions = [
        transactions_model.predict([[spend, #Today's Spend
                                     spend, #previous day's spend
                                     spend * 7, #spend over 7 days
                                     spend * month_days, #spend over 30 days
                                     visits_model.predict([[spend, #Today's Spend
                                                            spend, #previous day's spend
                                                            spend * 7, #spend over 7 days
                                                            spend * month_days, #spend over 30 days
                                                            ]])[0], # previous day's visits
                                     visits_model.predict([[spend, #Today's Spend
                                                            spend, #previous day's spend
                                                            spend * 7, #spend over 7 days
                                                            spend * month_days, #spend over 30 days
                                                            ]])[0]*7, #previous week's visit
                                     visits_model.predict([[spend, #Today's Spend
                                                            spend, #previous day's spend
                                                            spend * 7, #spend over 7 days
                                                            spend * month_days, #spend over 30 days
                                                            ]])[0]*month_days # previous month's visit                                                        
                                     ]])[0]
        for spend in spend_values
    ]

    cac_predictions = [
        spend / transactions if transactions > 0 else float('inf')
        for spend, transactions in zip(spend_values, transactions_predictions)
    ]

    transactions_df = pd.DataFrame({
        'Spend': spend_values,
        'Estimated Transactions': transactions_predictions,
        'Estimated CAC': cac_predictions
    })

    return transactions_df


def generate_campaign_csv(max_spend=outlier_cutoff, increment=100):
    all_campaigns_data = []

    campaign_df = generate_transactions_vs_spend( max_spend, increment)
    if campaign_df is not None:
        all_campaigns_data.append(campaign_df)

    final_df = pd.concat(all_campaigns_data, ignore_index=True)
    final_df.to_csv('overall_transactions_cac.csv', index=False)
    print("CSV saved as campaign_transactions_cac.csv")

    return final_df


def max_transactions_below_cac( cac_threshold=500, max_spend=outlier_cutoff, increment=100):
    campaign_df = generate_transactions_vs_spend( max_spend, increment)

    if campaign_df is None or campaign_df.empty:
        print(f"No data available for campaign")
        return {
            "Max Transactions": "Not enough data",
            "Spend": "Not enough data",
            "25th Percentile Transactions": "Not enough data",
            "50th Percentile Transactions": "Not enough data",
            "75th Percentile Transactions": "Not enough data"
        }

    filtered_df = campaign_df[campaign_df['Estimated CAC'] <= cac_threshold]

    if filtered_df.empty:
        print(f"No transactions found below CAC threshold for campaign:")
        return {
            "Max Transactions": "Not enough data",
            "Spend": "Not enough data",
            "25th Percentile Transactions": "Not enough data",
            "50th Percentile Transactions": "Not enough data",
            "75th Percentile Transactions": "Not enough data"
        }

    max_row = filtered_df.loc[filtered_df['Estimated Transactions'].idxmax()]

    percentiles = np.percentile(filtered_df['Estimated Transactions'], [25, 50, 75])

    return {
        "Max Transactions": max_row['Estimated Transactions'],
        "Spend": max_row['Spend'],
        "25th Percentile Transactions": percentiles[0],
        "50th Percentile Transactions": percentiles[1],
        "75th Percentile Transactions": percentiles[2],
        "CAC": max_row['Estimated CAC'],
        "25th Percentile CAC": max_row['Spend']/percentiles[0],
        "50th Percentile CAC": max_row['Spend']/percentiles[1],
        "75th Percentile CAC": max_row['Spend']/percentiles[2]
    }

def plot_metrics():
    data=pd.read_csv('overall_transactions_cac.csv')
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(
        data['Spend'], 
        data['Estimated Transactions'], 
        color=color, label="Transactions"
    )

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(
        data['Spend'], 
        data['Estimated CAC'], 
        color=color, label="CAC"
    )

    from textwrap import wrap

    ax1.set_title("\n".join(wrap("Overall Model: Predicted Transactions and CAC vs Spend",30)))
    ax1.set_ylabel("Transactions")
    ax2.set_ylabel("CAC")
    ax2.yaxis.set_major_formatter('${x:,.0f}')
    ax1.set_xlabel('Daily Spend')
    ax1.xaxis.set_major_formatter('${x:,.0f}')
    fig.autofmt_xdate(rotation=45)
    fig.legend(loc="upper left")
    ax2.axhline(y=500, color='r', linestyle='--')

    plt.savefig("plots/overall_predicted_transactions_and_cac_vs_spend.jpg")
    plt.show()



if __name__ == '__main__':
    transformed_df = transform_data(orig_df)

    transformed_df['date'] = pd.to_datetime(transformed_df['date'])

    campaigns_with_spend=transformed_df.groupby('campaign_name').agg({ 'spend':'sum'}).reset_index()
    campaigns_with_spend=campaigns_with_spend[campaigns_with_spend['spend']!=0]
    campaigns_with_spend=campaigns_with_spend['campaign_name'].unique()
    transformed_df = transformed_df[transformed_df['campaign_name'].isin(campaigns_with_spend)]

    results = {}
    feature_importances = {}
    models = {}

    # Code repurposed fromo Campaign models.py 
    # Move functions to utilities in later step
    campaign_data = transformed_df
    campaign_data=campaign_data[campaign_data['spend']<=outlier_cutoff]

    # This portion was used to balance data with high and low CAC over threshold. The models got worse, so this stratey was ababdond. 
    low_cac_data = campaign_data[campaign_data['spend'] / campaign_data['transactions'] <= 500]
    high_cac_data = campaign_data[campaign_data['spend'] / campaign_data['transactions'] > 500]

    balanced_campaign_data = pd.concat([
        low_cac_data,
        high_cac_data
    ])

    X_visits = balanced_campaign_data[['spend',
                                'previous_day_spend_campaign',
                                'previous_week_spend_campaign',
                                'previous_month_spend_campaign']]

    X_transactions = balanced_campaign_data[['spend',
                                'previous_day_spend_campaign',
                                'previous_week_spend_campaign',
                                'previous_month_spend_campaign',
                                'previous_day_visits_campaign',
                                'previous_week_visits_campaign',
                                'previous_month_visits_campaign'
                                ]]

    X_cac = balanced_campaign_data[['spend',
                                'previous_day_spend_campaign',
                                'previous_week_spend_campaign',
                                'previous_month_spend_campaign',
                                'previous_day_visits_campaign',
                                'previous_week_visits_campaign',
                                'previous_month_visits_campaign'
                                ]]
    y_transactions = balanced_campaign_data['transactions']
    y_visits = balanced_campaign_data['visits']
    y_cac = balanced_campaign_data['spend'] / balanced_campaign_data['transactions']


    X_train, X_test, y_train_t, y_test_t = train_test_split(X_transactions, y_transactions, test_size=0.2, random_state=42)
    rf_model_t = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_model_t.fit(X_train, y_train_t)
    y_pred_t = rf_model_t.predict(X_test)
    mse_t = mean_squared_error(y_test_t, y_pred_t)
    r2_t = r2_score(y_test_t, y_pred_t)



    X_train, X_test, y_train_v, y_test_v = train_test_split(X_visits, y_visits, test_size=0.2, random_state=42)
    rf_model_v = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_model_v.fit(X_train, y_train_v)
    y_pred_v = rf_model_v.predict(X_test)
    mse_v = mean_squared_error(y_test_v, y_pred_v)
    r2_v = r2_score(y_test_v, y_pred_v)

    X_train, X_test, y_train_c, y_test_c = train_test_split(X_cac, y_cac, test_size=0.2, random_state=42)
    rf_model_c = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_model_c.fit(X_train, y_train_c)
    y_pred_c = rf_model_c.predict(X_test)
    mse_c = mean_squared_error(y_test_c, y_pred_c)
    r2_c = r2_score(y_test_c, y_pred_c)


    result = {
    "Mean Squared Error (Transactions)": mse_t,
    "R-squared Score (Transactions)": r2_t,
    "Mean Squared Error (Visits)": mse_v,
    "R-squared Score (Visits)": r2_v,
    "Mean Squared Error (CAC)": mse_c,
    "R-squared Score (CAC)": r2_c
    }

    model = {
        'transactions': rf_model_t,
        'visits': rf_model_v,
        'cac': rf_model_c
    }


    feature_importance_df = pd.DataFrame({
        'Feature': X_cac.columns,
        'Importance (CAC)': rf_model_c.feature_importances_
    })
    feature_importance_df.to_csv(f'feature_importances/cac/overall_feature_importances.csv', index=False)

    feature_importance_df = pd.DataFrame({
        'Feature': X_visits.columns,
        'Importance (Visits)': rf_model_v.feature_importances_
    })
    feature_importance_df.to_csv(f'feature_importances/visits/overall_feature_importances.csv', index=False)

    feature_importance_df = pd.DataFrame({
        'Feature': X_transactions.columns,
        'Importance (Transactions)': rf_model_t.feature_importances_
    })
    feature_importance_df.to_csv(f'feature_importances/transactions/overall_feature_importances.csv', index=False)



    results_df = pd.DataFrame.from_dict(result, orient='index')
    results_df.to_csv('overall_model_results.csv', index=True)

    generate_campaign_csv(max_spend=outlier_cutoff, increment=100)
    campaign_results = []
    result = max_transactions_below_cac() 
    campaign_results.append(result)

    results_df = pd.DataFrame(campaign_results)
    results_df.to_csv('max_transactions_below_cac.csv', index=False)
    print("Results saved as max_transactions_below_cac.csv")
    print(results_df)
    plot_metrics()
