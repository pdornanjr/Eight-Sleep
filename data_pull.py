import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def pull_data(spend_campaigns_only=False):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name("sheets_creds.json", scope)
    gc = gspread.authorize(credentials)
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1d6NIVmPf9dF_qhTYvehl2AvfKeD3E-kEjXE522hwG9o/edit"
    sheet = gc.open_by_url(spreadsheet_url)
    worksheet = sheet.sheet1 
    records = worksheet.get_all_records()
    orig_df = pd.DataFrame(records)
    orig_df.to_csv("orig_df.csv")
    if spend_campaigns_only:
        df=orig_df[(orig_df['spend'] !=0)]
    else:
        df=orig_df
    df['date'] = pd.to_datetime(df['date'])
    df['week_number'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.isocalendar().year
    df['month_year'] = df['date'].dt.strftime('%Y-%m')
    return df


def get_data_by_month(df):
    basic_monthly_totals=df.groupby('month_year').agg({ 'spend':'sum',
                                                            'transactions':'sum',
                                                            'visits':'sum',
                                                            'rev':'sum'
                                                            }).reset_index()
    return basic_monthly_totals

def get_data_by_month_and_campaign(df):
    df_monthly=df.groupby(['month_year','campaign_name']).agg({ 'spend':'sum',
                                                            'transactions':'sum',
                                                            'visits':'sum',
                                                            'rev':'sum'
                                                            }).reset_index()
    return df_monthly


def add_metrics(df,groupby_cols):
    df['CAC']=df['spend']/df['transactions']
    df['spend_per_visit']=df['spend']/df['visits']
    df['percent_visits_with_transactions']=df['transactions']/df['visits']
    df['percent_daily_transactions']=df['transactions']/df.groupby(groupby_cols)['transactions'].transform('sum')
    df['percent_daily_spend']=df['spend']/df.groupby(groupby_cols)['spend'].transform('sum')
    df['percent_daily_visits']=df['visits']/df.groupby(groupby_cols)['visits'].transform('sum')
    df['percent_daily_rev']=df['rev']/df.groupby(groupby_cols)['rev'].transform('sum')
    df['transactions_index']=df['percent_daily_transactions']/df['percent_daily_spend']*100
    df['spend_index']=df['percent_daily_spend']/df['percent_daily_spend']*100
    df['visits_index']=df['percent_daily_visits']/df['percent_daily_spend']*100
    df['rev_index']=df['percent_daily_rev']/df['percent_daily_spend']*100

    return df


df=add_metrics(df=get_data_by_month_and_campaign(pull_data()),groupby_cols=['month_year'])
df.to_csv('monthly_with_metrics.csv')



def plot_cac_and_spend_by_month():
    basic_monthly_totals=get_data_by_month(pull_data())
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(
        basic_monthly_totals['month_year'], 
        basic_monthly_totals['spend'] / basic_monthly_totals['transactions'], 
        color=color, label="CAC", marker='o'
    )
    for x, y in zip(basic_monthly_totals['month_year'], 
                    basic_monthly_totals['spend'] / basic_monthly_totals['transactions']):
        ax1.text(x, y, f'${y:,.0f}', color=color, fontsize=8, ha='right', va='center', rotation=-45,
                transform=ax1.transData + plt.matplotlib.transforms.Affine2D().translate(5, 2))

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(
        basic_monthly_totals['month_year'], 
        basic_monthly_totals['spend'], 
        color=color, label="Spend", marker='o'
    )
    for x, y in zip(basic_monthly_totals['month_year'], basic_monthly_totals['spend']):
        ax2.text(x, y, f'${y:,.0f}', color=color, fontsize=8, ha='left',va='center',rotation=-45,
                transform=ax2.transData + plt.matplotlib.transforms.Affine2D().translate(-5, -2))

    fig.tight_layout() 
    plt.title("Monthly Spend and CAC")
    ax1.set_ylabel("CAC")
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax2.set_ylabel("Monthly Spend")
    ax2.yaxis.set_major_formatter('${x:,.0f}')
    ax1.set_xlabel('Month/Year')
    fig.autofmt_xdate(rotation=45)
    fig.legend(loc="upper left")
    plt.savefig(f"plots/Monthly Spend and CAC.jpg")

    return print("Plot Saved in Plots Folder")

