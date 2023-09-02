#!/usr/bin/env python
# coding: utf-8

# In[19]:


from bs4 import BeautifulSoup as soup 
import requests
page = requests.get("http://192.168.0.188:8080/employee/content/")
parse = soup(page.content,'html.parser')

htmltable = parse.find('table', { 'class' : 'table table-striped small' })


# In[20]:


import pandas as pd
df = pd.read_html(str(htmltable))[0]


# In[21]:


from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px


# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Virtualisation Assignment 1"),
    dcc.Graph(figure=px.histogram(df, x='country', y='age', histfunc='avg')),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
])

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)


# In[ ]:




