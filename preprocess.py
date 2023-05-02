import pandas as pd
import numpy as np


def see_col_counts():  # use to understand data a bit better
    x = pd.read_csv('train_x.csv')
    x.drop(columns=['Unnamed: 0'], inplace=True)
    test_cols = ['cellattributevalue', 'celllabel', 'nursingchartcelltypevalname', 'nursingchartvalue', 'labmeasurenamesystem', 'labname', 'labresult']
    # test_cols = ['admissionheight', 'admissionweight', 'age', 'ethnicity', 'unitvisitnumber']
    print((~(x[test_cols].isna())).sum(axis=0))
    for col in test_cols:
        print(col)
        print('-' * 30)
        print(x[col].value_counts(dropna=True))
        print()
    
def fix_age(df):
    df['age'] = df['age'].apply(lambda x : 90 if x == '> 89' else x)

def test_pivot():  # testing pandas pivot function
    x = pd.read_csv('train_x.csv')
    x.drop(columns=['Unnamed: 0'], inplace=True)
    x.drop(columns=['celllabel', 'labmeasurenamesystem'], inplace=True)
    fix_age(x)

    # print(x[~(x['nursingchartcelltypevalname'].isna())]['patientunitstayid'].iloc[0])
    # t = x[x['patientunitstayid'] == 1754323][['labname', 'labresult']]
    # t.iloc[t.shape[0] - 6, 0] = 'pH'
    # t.iloc[t.shape[0] - 6, 1] = 1
    # print(t)
    # t = t.pivot(columns='labname', values='labresult')
    # t = t.drop(columns=[t.columns[0]])
    # print(t.mean(axis=0))

    chart_types = x['nursingchartcelltypevalname']
    chart_types = chart_types[~chart_types.isna()].unique()
    print(chart_types)
    df = pd.DataFrame(columns=chart_types)
    print(df)
    t = x[x['patientunitstayid'] == 143870][['nursingchartcelltypevalname', 'nursingchartvalue']]
    t = t.pivot(columns='nursingchartcelltypevalname', values='nursingchartvalue')
    t = t.drop(columns=[t.columns[0]])
    t = t.mean(axis=0)
    print(t)
    df.loc[0, t.index] = t
    print(df)

def global_replacement_vals(x, lab_types, chart_types):
    lab_std_cols = [col for col in lab_types if col[-3:] == 'std']
    chart_std_cols = [col for col in chart_types if col[-3:] == 'std']
    count_cols = [col for col in lab_types + chart_types if col[-5:] == 'count']

    numeric = ['admissionheight', 'admissionweight', 'age', 'offset', 'unitvisitnumber']
    tests = lab_types + chart_types
    text = ['cellattributevalue', 'ethnicity', 'gender']
    replacement_vals = pd.DataFrame(columns=(numeric + tests + text))
    replacement_vals.loc[0, count_cols] = 0  # for replacement values for count columns, 0 count

    labs = x[['labname', 'labresult']]
    labs = labs.pivot(columns='labname', values='labresult')
    labs = labs.drop(columns=[labs.columns[0]])
    labs_mean = labs.mean(axis=0)
    labs_std = labs.std(axis=0)
    labs_std.index = [idx + '_std' for idx in labs_std.index]
    replacement_vals.loc[0, labs_mean.index] = labs_mean
    replacement_vals.loc[0, labs_std.index] = labs_std

    charts = x[['nursingchartcelltypevalname', 'nursingchartvalue']]
    charts = charts.pivot(columns='nursingchartcelltypevalname', values='nursingchartvalue')
    charts = charts.drop(columns=[charts.columns[0]])
    charts_mean = charts.mean(axis=0)
    charts_std = charts.std(axis=0)
    charts_std.index = [idx + '_std' for idx in charts_std.index]
    replacement_vals.loc[0, charts_mean.index] = charts_mean
    replacement_vals.loc[0, charts_std.index] = charts_std

    replacement_vals.loc[0, numeric] = x[numeric].astype(float).mean(axis=0)
    replacement_vals.loc[0, text] = 'none'
    return replacement_vals

def process_patient(x, id, processed_df, chart_types, lab_types, replacement_vals):
    row_idx = len(processed_df.index)
    patient = x[x['patientunitstayid'] == id]
    # non_test_cols = patient[['admissionheight', 'admissionweight', 'age', 'cellattributevalue', 'ethnicity', 'gender', 'offset', 'unitvisitnumber']]
    test_cols = chart_types + lab_types
    numeric = ['admissionheight', 'admissionweight', 'age', 'offset', 'unitvisitnumber']
    text = ['cellattributevalue', 'ethnicity', 'gender']
    lab_cols = patient[['labname', 'labresult']]
    nurse_cols = patient[['nursingchartcelltypevalname', 'nursingchartvalue']]

    # COMPUTE IN PREPROCESS - GLOBAL AVG/REPLACEMENT VALUES FOR EACH COLUMN AS A DATAFRAME
    # consider when nothing is returned
    lab_cols = lab_cols.pivot(columns='labname', values='labresult')
    lab_cols = lab_cols.drop(columns=[lab_cols.columns[0]])
    if lab_cols.size != 0:
        lab_cols_mean = lab_cols.mean(axis=0)
        lab_cols_std = lab_cols.std(axis=0)
        lab_cols_std.index = [idx + '_std' for idx in lab_cols_std.index]
        lab_cols_count = (~lab_cols.isna()).sum(axis=0)
        lab_cols_count.index = [idx + '_count' for idx in lab_cols_count.index]
        processed_df.loc[row_idx, lab_cols_mean.index] = lab_cols_mean
        processed_df.loc[row_idx, lab_cols_std.index] = lab_cols_std
        processed_df.loc[row_idx, lab_cols_count.index] = lab_cols_count

    nurse_cols = nurse_cols.pivot(columns='nursingchartcelltypevalname', values='nursingchartvalue')
    nurse_cols = nurse_cols.drop(columns=[nurse_cols.columns[0]])
    if nurse_cols.size != 0:
        nurse_cols_mean = nurse_cols.mean(axis=0)
        nurse_cols_std = nurse_cols.std(axis=0)
        nurse_cols_std.index = [idx + '_std' for idx in nurse_cols_std.index]
        nurse_cols_count = (~nurse_cols.isna()).sum(axis=0)
        nurse_cols_count.index = [idx + '_count' for idx in nurse_cols_count.index]
        processed_df.loc[row_idx, nurse_cols_mean.index] = nurse_cols_mean
        processed_df.loc[row_idx, nurse_cols_std.index] = nurse_cols_std
        processed_df.loc[row_idx, nurse_cols_count.index] = nurse_cols_count
    
    num = patient[numeric].astype(float).mean(axis=0)
    # think first row has all text data for a patient, but backfill just in case and basically take the first occurrence of ethnicity, gender, cellattributevalue
    # which are basically unique for a patient
    txt = patient[text].fillna(method='bfill').iloc[0]
    txt.name = None

    num[num.isna()] = replacement_vals[num[num.isna()].index].iloc[0]
    txt[txt.isna()] = replacement_vals[txt[txt.isna()].index].iloc[0]
    assert (~num.isna()).all() and (~txt.isna()).all()

    processed_df.loc[row_idx, 'patientunitstayid'] = id
    processed_df.loc[row_idx, num.index] = num
    processed_df.loc[row_idx, txt.index] = txt
    processed_df.loc[row_idx, test_cols] = processed_df.loc[row_idx, test_cols].where(~processed_df.loc[row_idx, test_cols].isna(), replacement_vals[test_cols].iloc[0])
    return processed_df

def preprocess():
    pd.set_option('display.max_columns', 50)

    x = pd.read_csv('train_x.csv')
    x.drop(columns=['Unnamed: 0'], inplace=True)
    # probably drop celllabel - all capillary refill, so only the cellattributevalue matters
    # labmeasurenamesystem is all mg/dL, can drop that for the labresult
    # labname is either glucose test or pH test, reported in mg/dL
    x.drop(columns=['celllabel', 'labmeasurenamesystem'], inplace=True)
    fix_age(x)
    x['nursingchartvalue'] = x['nursingchartvalue'].apply(lambda z : np.nan if type(z) == str else z).astype(float)

    y = pd.read_csv('train_y.csv')
    y.drop(columns=['Unnamed: 0'], inplace=True)

    chart_types = x['nursingchartcelltypevalname']
    chart_types = list(chart_types[~chart_types.isna()].unique())

    lab_types = ['pH', 'glucose']

    # add standard deviation and count columns for each reading, because there can be multiple of these for a single patient
    tmp  = []
    for chart_type in chart_types:
        tmp.append(chart_type)
        tmp.append(chart_type + '_std')
        tmp.append(chart_type + '_count')
    chart_types = tmp
    tmp = []
    for lab_type in lab_types:
        tmp.append(lab_type)
        tmp.append(lab_type + '_std')
        tmp.append(lab_type + '_count')
    lab_types = tmp

    replacement_vals = global_replacement_vals(x, lab_types, chart_types)

    processed_cols = ['admissionheight', 'admissionweight', 'age', 'cellattributevalue', 'ethnicity', 'gender', 'offset', 'unitvisitnumber']
    processed_cols += chart_types
    processed_cols += lab_types
    processed_cols += ['patientunitstayid']
    processed_df = pd.DataFrame(columns=processed_cols)

    for id in y['patientunitstayid']:
        process_patient(x, id, processed_df, chart_types, lab_types, replacement_vals)

    return processed_df

if __name__ == '__main__':
    # to understand data, just look at all rows for a specific patient
    # pd.options.display.max_columns = 50
    # see_col_counts()
    
    # columns = [admissionheight, admissionweight, age, cellattributevalue, ethnicity, gender, labname, labresult, 
    #            nursingchartcelltypevalname, nursingchartvalue, offset, patientunitstayid, unitvisitnumber]
    # test_pivot()

    # add column for patient id
    processed_df = preprocess()
    processed_df.to_csv('processed_train_x_v2.csv')

