#!/usr/bin/env python
# coding: utf-8

# In[7]


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime, timedelta

'''
=====================================================
Helpers functions for removing days without PV output
=====================================================
'''

# To filter only day time
def sun_hours(day):
    sun_rise = datetime(year=day.year, month=day.month, day=day.day,
                     hour=5, minute=30)
    sun_set = sun_rise + timedelta(hours=13)
    return sun_rise, sun_set

# To check PV output power in one day 
def check_PV_power(sun_rise, sun_set):
    a_day = original_df.loc[sun_rise:sun_set]
    is_zero = a_day.apply(check_zero_PV, axis=1)
    count_zero = is_zero.value_counts()
    if len(list(count_zero.index)) == 1:
        return sun_rise.strftime('%x') if list(count_zero.index)[0] else None
    else:
        zero_PV_percentage = count_zero.loc[True] / len(a_day)
        return sun_rise.strftime('%x') if zero_PV_percentage >= 0.20 else None

    
# helper function to use with apply method
def check_zero_PV(x):
    if x['solar_rad'] > 0:
        return True if x['PV_Power'] <= 0 else False
    else:
        return False
    

#####################################################################

# The main function to call for filter days with no PV output power
def filter_produced_day(df):
    current_pointer = df.index[0]
    next_day = timedelta(days=1)
    non_produced_day = []
    
    while current_pointer <= df.index[-1]:
        sun_rise, sun_set = sun_hours(current_pointer)
        no_power = check_PV_power(sun_rise, sun_set)
        if no_power:
            non_produced_day.append(no_power)        
        current_pointer += next_day
    
    selected_indexs = list(map(lambda x : x not in non_produced_day, df.index.strftime('%x')))
    new_df = df.loc[df.index[selected_indexs]]
    
    print(f"There are {len(non_produced_day)} days which don't have PV output power. \n"
          f'Including: {non_produced_day} \n')
    print(f"Total samples remaining = {len(new_df)} \n" 
          f"Total samples originally = {len(df)}")
    print("The total days with no PV power is {perc:.2f} % of overall data".format(perc = ((len(df) - len(new_df)) / len(df)) * 100))
    
    return new_df


# # Calculate Sun Position 

# In[9]:


'''
===========================================
Helper functions for calculating each angle
===========================================
'''

def de_to_rad(x):
    return x * np.pi / 180.    

def declination_angle(x):
    radians_angle = de_to_rad(((360/365)*(x.name.dayofyear - 81))) 
    return de_to_rad(23.45*np.sin(radians_angle))  

def altitude_angle(x):
    # Solomon Location = 13.903, 100.53
    lat, long = 13.903, 100.53

    hour_angle = 15 * (12 - (x.name.hour + (x.name.minute / 60)))
    sine_angle = (np.cos(de_to_rad(lat)) * np.cos(x['declination_angle']) 
                  * np.cos(de_to_rad(hour_angle))) + (np.sin(de_to_rad(lat)) * np.sin(x['declination_angle']))
    return np.arcsin(sine_angle)

def azimuth_angle(x):
    hour_angle = 15 * (12 - (x.name.hour + (x.name.minute / 60)))
    sine_angle = (np.cos(x['declination_angle']) * np.sin(de_to_rad(hour_angle))) / np.cos(x['altitude_angle'])
    return np.arcsin(sine_angle)
    


# # Split X, y

# In[10]:


'''
===========================================
Helper functions for spliting X, y 
===========================================
'''

# A helper function to split dataframe into X and y
def split_X_y(df, ignored_cols):
    '''
    Split data into X and y
    '''
    y_label = 'PV_Power'
    
    # `selected_cols` is a list of 'True' or 'False' defined by lambda function
    selected_cols = list(map(lambda x: x not in (ignored_cols + [y_label]), df.columns)) 
    
    X, y = df[df.columns[selected_cols]], df[y_label]
    return X.astype('float32'), y.astype('float32')  

# A helper function to split `categorical` and `numerical` columns
def split_num_cat(df):
    '''
    Split `categorical` and `numerical` columns into two lists
    '''
    categorical_columns = []
    numerical_columns = []

    # find the categorical columns
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            categorical_columns.append(label)
        else:
            numerical_columns.append(label)
    
    return categorical_columns, numerical_columns


# # Extract weather description

# In[11]:


'''
===============================================
Helper function to extract weather description
===============================================
'''

def extract_weather_des(df):
    df['weather_icon'] = df['weather'].str.extract(r"'icon': '(.*?)',", expand=False)
    df['weather_code'] = df['weather'].str.extract(r"'code': (.*?),", expand=False)
    df['weather_description'] = df['weather'].str.extract(r"'description': '(.*?)'", expand=False)
    return df

'''
===========================================
Helper functions for evaluation metrics
===========================================
'''

def RMSE(y_test, y_preds):
    '''
    Calculate root mean squared error between predictions and the targets
    '''
    return np.sqrt(mean_squared_error(y_test, y_preds))

def MBE(y_test, y_preds):
    '''
    Calculate mean bias error between predictions and the targets
    '''
    return (y_preds - y_test).sum() / len(y_test)

def APE(y_pred, y_true):
    '''
    Calculate Absolute Percentage Error (APE)
    '''
    error = np.absolute(y_pred - y_true)
    return np.round((error*100 / y_true), 2)    

def forecasting_result(model, X_test):
    '''
    Get forecast output power for using with plot function
    '''
    return np.squeeze(model.predict(X_test)) 

def show_scores(y_preds, y_test): 
    '''
    Calculate and show all evaluated metrics 
    '''
    
    peak_power = 10000 # Peak or installed power of the PV system
    avg_power = np.mean(y_test) # average PV output power from dataset using PV value 

    scores = {'MAE': np.round(mean_absolute_error(y_test, y_preds), 2),
              'RMSE': np.round(RMSE(y_test, y_preds), 2),
              'MBE' : np.round(MBE(y_test, y_preds), 2),
              'NMAE_installed': np.round((mean_absolute_error(y_test, y_preds) / peak_power), 4) * 100,
              'NRMSE_installed': np.round((RMSE(y_test, y_preds) / peak_power), 4) * 100,
              'NMAE_avg' : np.round((mean_absolute_error(y_test, y_preds) / avg_power), 4) * 100,
              'NRMSE_avg' : np.round((RMSE(y_test, y_preds) / avg_power), 4) *100
             }    
    
    print(("Mean Absolute Error = {MAE:.2f} W \n").format(MAE = scores['MAE']) +
         ("Root Mean Squared Error = {RMSE:.2f} W \n").format(RMSE = scores['RMSE']) +
         ("Mean Bias Error = {MBE:.2f} W \n \n").format(MBE = scores['MBE']) +
         ("Normalized_MAE (installed_Power) = {NMAE:.2f}% \n").format(NMAE = scores['NMAE_installed']) +
         ("Normalized_RMSE (installed_Power) = {NRMSE:.2f}% \n \n").format(NRMSE = scores['NRMSE_installed']) +
         ("Normalized_MAE (avg_Power) = {NMAE:.2f}% \n").format(NMAE = scores['NMAE_avg']) +
         ("Normalized_RMSE (avg_Power) = {NRMSE:.2f}% \n \n").format(NRMSE = scores['NRMSE_avg']) +
         ("Peak PV output power (from y_test) = {peak_power:.2f} W \n").format(peak_power = np.max(y_test)) +
         ("Average PV output power (from y_test) = {avg_power:.2f} W").format(avg_power = avg_power))
    
    return scores    

'''
===========================================
Helper functions for calculating kWh (energy)
===========================================
'''
def plot_histogram(the_data, bin_width=10, max_range=100):
    '''
    Plot histogram of absolute percentage error of input data
    '''
    sns.histplot(data=the_data, stat='proportion', binwidth=bin_width, binrange=(0, max_range))
    plt.xlabel('Absolute Percentage Error (%)')
    plt.axvline(the_data.mean(), color='orange', label='Mean', linewidth=3)
    plt.axvline(the_data.median(), color='green', label='Median', linewidth=3)
    plt.legend();    

def fill_na_inf(df, fixed_percent=100):
    '''
    Fill NaN with 0 and fill Infinite with fixed_percent (default = 100)    
    '''
    return df.fillna(0).replace([np.inf, -np.inf], fixed_percent)


def daily_kWh(y, x):
    '''
    Calculate daily kWh (calculate area under Power vs Time curve)
    '''
    kWh = float(np.trapz(y, x))
    return np.round(kWh / 3.6e+15, 3) 

import datetime
def calculating_kWh(y_true, y_preds):
    '''
    Calculate energy (kWh) from actual power (y_true) and
    predict power (y_preds) and put them in a dataframe for futher analysis. 
    
    The function takes two arguments (y_true and y_preds) and returns a dataframe 
    contained energy (kWh) of actual power (y_true) and predict power (y_preds)  
    '''

    # concat y_true and y_preds as one dataframe
    both_y = pd.concat([y_true.reset_index(), y_preds], axis=1)
    both_y = both_y.set_index('timestamp_local')
    both_y.columns = ['y_true', 'y_preds']
    
    # Filter only daytime
    both_y = both_y[(both_y.index.time >= datetime.time(6,0)) & (both_y.index.time <= datetime.time(18,0))]
    
    # Check total day in `both_y` data frame
    day_ts = both_y.resample('D').size().index.date
    
    # Calculate daily kWh
    y_true_kWh = []
    y_preds_kWh = []
    for day in day_ts:
        y_true_kWh.append(daily_kWh(both_y[both_y.index.date == day]['y_true'], both_y[both_y.index.date == day].index))
        y_preds_kWh.append(daily_kWh(both_y[both_y.index.date == day]['y_preds'], both_y[both_y.index.date == day].index))
    
    return pd.DataFrame({'y_true_kWh':y_true_kWh, 'y_preds_kWh':y_preds_kWh}, index=pd.to_datetime(day_ts)).query('y_true_kWh > 0')


def plot_kWh(df):
    '''
    Plot daily kWh of the first 10 day 
    '''
    plt.bar(df[:10].index, df[:10]['y_test_kWh'], label='y_test')
    plt.bar(df[:10].index, df[:10]['y_preds_kWh'], label='y_preds')
    plt.legend(['y_test', 'y_preds'])
    plt.xlabel('date')
    plt.xticks(rotation=90)
    plt.ylabel('Energy (kWh)')
    plt.title('Actual energy vs Predict energy')
    plt.show()


'''
=========================================
Helper functions for deep learning models
=========================================
'''

########################## callable functions #############################

# To plot model loss and accuracy vs epochs 
def loss_acc_epochs(history, model_name = '', loss_name = ''):
    sns.set_style('whitegrid')
    history_df = pd.DataFrame(history.history)
    epcohs = history_df.index + 1
    steps = 20 if epcohs[-1] > 40 else 1
    
    
    # Loss 
    plt.figure(figsize=(15,4))
    sns.lineplot(data=history_df, y='loss', x=epcohs, label='loss')
    sns.lineplot(data=history_df, y='val_loss', x=epcohs, label='val_loss')
    plt.legend()
    plt.title(f'{model_name}: Loss', fontsize=14)
    plt.xlabel('EPOCHs')
    plt.xticks(range(1, epcohs[-1] + 1, steps))
    plt.ylabel(f'Loss: {loss_name}', fontsize=12);
    
    # Accuracy
    if len(pd.Series(history_df.columns.isin(['accuracy'])).unique()) == 2:
        plt.figure(figsize=(15,4))
        sns.lineplot(data=history_df, y='accuracy', x=epcohs, label='accuracy')
        sns.lineplot(data=history_df, y='val_accuracy', x=epcohs, label='val_accuracy')
        plt.legend()
        plt.title(f'{model_name}: Accuracy', fontsize=14)
        plt.xlabel('EPOCHs')
        plt.xticks(range(1, epcohs[-1] + 1, steps))
        plt.ylabel('Accuracy', fontsize=12);

        
# Create a function to visualize and compare PV output power (between forecasting and validation data)
def plot_pv(y_preds, y_test, model_name, plt_range):
    
    # Create df for forecast and actual PV power
    y = pd.DataFrame({'Forecast_power':y_preds, 'Actual_power':y_test})
    
    # Line plot to compare forecast and actual PV power
    plt.figure(figsize=(15,8))
    
    y['Forecast_power'].loc[f"{plt_range['year']}-{plt_range['month']}-{plt_range['start']}"
          :f"{plt_range['year']}-{plt_range['month']}-{plt_range['end']}"].plot(label='Forecast power')
    y['Actual_power'].loc[f"{plt_range['year']}-{plt_range['month']}-{plt_range['start']}"
          :f"{plt_range['year']}-{plt_range['month']}-{plt_range['end']}"].plot(label='Actual power')

    plt.ylabel('Output Power')
    plt.title(f'Forecast vs Actual power - {model_name} forecasting')
    plt.legend()
    plt.show()


'''
===============================================================
Helper function to save the best model during training
===============================================================
'''
def model_checkpoint(model_name):
    # Create log directory
    log_dir = f'saved_models/{model_name}' 
    
    # Create model checkpoint
    checkpoint = ModelCheckpoint(filepath=log_dir, save_weights_only=False, 
                                 monitor='val_loss', mode='auto',
                                 save_best_only=True, verbose=0)
    print(f'The model is saved in "{log_dir}"')
    return checkpoint

def early_stop(check='val_loss', wait=3):
    es = EarlyStopping(monitor=check, patience=wait)
    return es