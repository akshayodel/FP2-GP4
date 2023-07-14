#!/usr/bin/env python
# coding: utf-8

# # Oil Production Forecasting

# In[1]:


#importing libraries

import json
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings #library to ignore warnings
warnings.filterwarnings('ignore')
import ipywidgets as widgets #widgets for voila deployment
from IPython.display import display, clear_output

from dateutil.relativedelta import relativedelta

from datetime import datetime


# In[2]:


# API URL
api_url="https://api.eia.gov/v2/international/data/?api_key=PYTvNry19FP8DgMdBAikT14IOtP21vBOZY2ZuDTJ&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=53&facets[countryRegionId][]=IND&facets[unit][]=TBPD&start=1990-01&end=2023-03&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"
# Send GET request to the API
response = requests.get(api_url)


# In[3]:


# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the response data
    data = response.json()
    df = pd.DataFrame(data['response']['data'])
    #print("Data extracted successfully via API.")
#else:
    #print("Failed to retrieve data. Status code:", response.status_code)


# In[4]:


# Convert 'period' column to datetime format and assign it to 'ds' column
df['ds']=pd.to_datetime(df['period'])

# Assign 'value' column to 'y' column
df['y']=df['value']


# In[5]:


# Create a new DataFrame with columns 'ds' and 'y'
data_series=df[['ds','y']]


# In[6]:


import pandas as pd

API_TOKEN = 'a122af78b08f7eccfe99605fe17fb2050d3116e1'

# Read CPI data from the specified URL and parse the 'Date' column as dates
df_cpi = pd.read_csv(
    'https://www.econdb.com/api/series/CPIIN/?token=%s&format=csv' % API_TOKEN,
    parse_dates=['Date'])


# In[7]:


# Merge the CPI dataframe (df_cpi) and the data series dataframe (data_series)
# based on the common columns 'Date' and 'ds', using an inner join

merged_df = df_cpi.merge(data_series, left_on='Date', right_on='ds', how='inner')


# In[8]:


# Import the ARIMA model from the statsmodels library for time series analysis and forecasting

from statsmodels.tsa.arima.model import ARIMA


# In[9]:


#Defining period

period=len(merged_df) - 6


# In[10]:


# Select the columns 'ds', 'CPIIN', and 'y' from the merged DataFrame
# and slice the rows from the beginning up to 'period' (exclusive) to create the training data

train_data =merged_df[['ds', 'CPIIN','y']][:period]

# Select the columns 'ds', 'CPIIN', and 'y' from the merged DataFrame
# and slice the rows from the beginning up to 'period' (exclusive) to create the training data
test_data =merged_df[['ds', 'CPIIN','y']][period:]


# In[11]:


order = (1, 0, 0)  # ARIMA order

# Create an ARIMAX model with the specified order, using 'y' as the endogenous variable
# and 'CPIIN' as the exogenous variable in the training data
model_arimax = ARIMA(train_data['y'], exog=train_data['CPIIN'], order=order)

# Fit the ARIMAX model to the training data
model_arimax_fit = model_arimax.fit()


# In[12]:


#Predicting oil production
exog_reshaped = test_data['CPIIN'].values.reshape(-1, 1)
# Make predictions using the trained ARIMAX model
predicted_values_user = model_arimax_fit.predict(start=len(merged_df)-6, end=len(merged_df)-6 + 6 - 1, exog=exog_reshaped)

# Calculate MAPE
actual_values = test_data['y']
mape = round(np.mean(np.abs((actual_values - predicted_values_user) / actual_values)),4)


# In[13]:


# Calculate MAPE row-wise
mape_list = []
for i in range(357,363):
    actual_value_1 = actual_values[i]
    predicted_value_1 = predicted_values_user[i]
    mape = round(np.abs((actual_value_1 - predicted_value_1) / actual_value_1), 4)
    mape_list.append(mape)

# Add the MAPE values as a new column to the DataFrame
test_data['MAPE'] = mape_list

Monitoring_data = test_data[['ds','MAPE']]
Monitoring_data.reset_index(drop=True, inplace=True)
Monitoring_data.to_csv('monitoring_data.csv',index=False)


# In[14]:


style = {'description_width': 'initial'}

# Create a date widget
start_date = widgets.DatePicker(
    description='Select Test Start Date',
    disabled=True,
    style=style
    ,value=merged_df['ds'].dt.date.max() - relativedelta(months=5)
)


# Create a date widget
end_date = widgets.DatePicker(
    description='Select Test End Date',
    disabled=False,
    style=style
    ,value=datetime.today().date() + relativedelta(months=4)
)


# In[ ]:





# In[15]:


# ARIMA to predict CPIIN

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
# Create a DataFrame with the date and value columns
data = merged_df[:period]
# Set the date column as the index
data.set_index('ds', inplace=True)
# Create the ARIMA model
model = ARIMA(data['CPIIN'], order=(1, 0, 0))  # Set the order (p, d, q) for ARIMA
# Fit the model
model_fit = model.fit()


# In[16]:


# Voila


# In[17]:


#Install these packages for the first time users

#!pip install voila
#!pip install widgetsnbextension
#!pip uninstall ipywidgets --only in terminal
#!pip install ipywidgets
get_ipython().system('jupyter nbextension enable --py widgetsnbextension --sys-prefix')
get_ipython().system('jupyter serverextension enable voila --sys-prefix')


# In[18]:


def prediction(months_diff,Title):
    # Predicting CPIIN
    predicted_value_cpiin_user = pd.DataFrame(model_fit.predict(start=len(data), end=len(data) + months_diff - 1))
    predicted_value_cpiin_user['CPIIN'] = predicted_value_cpiin_user['predicted_mean']
    # Create a new index
    new_index = range(len(data), len(data) + months_diff)
    # Set the new index using set_index()
    predicted_value_cpiin_user.index = new_index

    #Predicting oil production
    exog_reshaped = predicted_value_cpiin_user['CPIIN'].values.reshape(-1, 1)
    # Make predictions using the trained ARIMAX model
    predicted_values_user = pd.DataFrame(model_arimax_fit.predict(start=len(merged_df)-6, end=len(merged_df)-6 + months_diff - 1, exog=exog_reshaped))
    predicted_values_user['Oil_Production'] = predicted_values_user['predicted_mean'].round(2)
    # Reset the index and start from 0
    predicted_values_user.reset_index(drop=True, inplace=True)

    # Generate the monthly dates within the specified range    
    dates = pd.date_range(start=start_date.value, end=end_date.value, freq='MS')
    # Create a DataFrame with the month column
    predicted_values_user['Month'] = pd.DataFrame({'Month': dates})

    predicted_values_user = predicted_values_user.drop(columns=["predicted_mean"])
    predicted_values_user = predicted_values_user[['Month','Oil_Production']]
    
    print(predicted_values_user.to_string(index=False))

    # Plot the line graph
    predicted_values_user.plot(x='Month', y='Oil_Production', figsize=(8, 4))

    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Oil_Production')
    plt.title('Oil Production Forecast '+Title)

    # Show the plot
    plt.show()


# In[19]:


button_send_2 = widgets.Button(
                description='Predict',
                tooltip='Send',
                style={'description_width': 'initial'}
            )

output_2 = widgets.Output()

def on_button_clicked(event):
    with output_2:          
        # Calculate the difference between the end date and start date
        date_diff = end_date.value - start_date.value
       
        # Calculate the difference in months between the end date and start date
        months_diff = (end_date.value.year - start_date.value.year) * 12 + (end_date.value.month - start_date.value.month) + 1
        
        # Calculate the difference in months between the end date and start date
        Short_Term_Duration = 6
        Long_Term_Duration = 60
        
        print("Oil Production Forecast: Custom Date")
        prediction(months_diff, "Oil Production Forecast: Custom Date")
        
        # Calculate the difference in months between the end date and start date
        end_date.value = start_date.value + relativedelta(months=Short_Term_Duration)
        print("Oil Production Forecast: Short-Term Duration")
        prediction(Short_Term_Duration, "Oil Production Forecast: Short-Term Duration")
        
        # Calculate the difference in months between the end date and start date
        end_date.value = start_date.value + relativedelta(months=Long_Term_Duration)
        print("Oil Production Forecast: Long-Term Duration")
        prediction(Long_Term_Duration, "Oil Production Forecast: Long-Term Duration")
        
        # Reset the end date to the original difference in days
        end_date.value = start_date.value + relativedelta(days=date_diff.days)
        

button_send_2.on_click(on_button_clicked)

text_0 = widgets.HTML(value="<h1>Select the end date</h1>")
vbox_result = widgets.VBox([text_0, output_2,start_date,end_date,button_send_2])


# In[20]:


page = widgets.HBox([vbox_result])

# Display the page containing the result VBox
display(page)


# In[ ]:




