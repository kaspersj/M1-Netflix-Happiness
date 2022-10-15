import streamlit as st

#################### Imports: ############################
## Importing the usual suspects
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd

#Matplotlib library
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# For date time conversion
import time
from datetime import datetime, timedelta
import pickle
import folium

# From Altair
import altair as alt
import altair_ally as aly

#From Plotly
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# %reload_ext watermark

############################ Disabling Maxrows ##############################
alt.data_transformers.disable_max_rows()
################## Title and beginning of analysis ###################

st.title('Our Netflix Analysis')

st.header("John Kaspers (kaspersj), Ong Hock Boon Steven David (steveong), Chi Huen Fong (chfong)")
st.markdown('\n\n')
st.header("Supplementary visualizations - some are interactive!")
st.markdown('\n\n')

# Data Sources
# Netflix Pricing
# https://www.comparitech.com/blog/vpn-privacy/countries-netflix-cost/
# World Happiness Report
# https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021
# World Development Indicators
# https://datatopics.worldbank.org/world-development-indicators/

###########################################################################

# We export the final dataframe as df.csv
df = pd.read_csv('df.csv')
st.dataframe(df)

# Drop first column
df = df.iloc[: , 1:]
df.head()

happiness_years = pd.read_csv('Happiness.csv')
happiness_years.head()

# Drop first column
happiness_years = happiness_years.iloc[: , 1:]
happiness_years.head()

region_grouped = df.groupby(['Region']).agg('mean')
region_grouped


########### Visualizing df ##########
aly.dist(df, mark='bar')

g = sns.PairGrid(df, palette='rainbow', hue="Region")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

####### Preparing the Happiness dataframe

# Drop NaN in happiness score
cleaned_df = df.dropna(subset=["Happiness score"])
cleaned_df.head()

############ Happiness score chart by country ####
st.markdown('Happiness Score Chart')
country_happiness_chart = alt.Chart(cleaned_df).mark_bar().encode(
    x = alt.X('Happiness score:Q'),
    y = alt.Y('Country:N', sort='-x')
)
st.altair_chart(country_happiness_chart)
st.markdown('\n\n')
st.markdown('Regional Happiness Score Chart')
###### Regional happiness
happiness_score_by_country_chart = alt.Chart(cleaned_df).mark_bar().encode(
    x = alt.X('Happiness score:Q'),
    y = alt.Y('Region:N',sort='-x')
)

st.altair_chart(happiness_score_by_country_chart)
################# Price of Netflix Worldwide
fig = px.choropleth(df, 
                        locations="Country", locationmode='country names',
                    color="Netflix Price",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Jet,
                    labels= {'Netflix Price': 'Netflix Price'},
                    title = 'Netflix Prices Worldwide 2021')

fig.update(layout=dict(title=dict(x=0.5)))
# fig =  plt.subplots()
# fig.show()
st.plotly_chart(fig)

##################### Happiness scores over time
fig = px.choropleth(happiness_years, locations="Country", locationmode='country names',
                    color="Happiness Score",
                    hover_name="Country",
                    color_continuous_scale=[[0, 'rgb(255, 0, 0)'],#red
                      [0.25, 'rgb(255,165,0)'], #orange
                      [0.5, 'rgb(255, 255, 0)'],#yellow
                      [0.75, 'rgb(24,252,0)'], # green
                      [1, 'rgb(0,100,10)']], # dark green
                    animation_frame= "Year", range_color = [0,8],
                    #labels= {'Happiness Score': 'Happiness Score 2015'},
                    title = 'Happiness Score - 2015 to 2021',
                    
                    )

fig.update(layout=dict(title=dict(x=0.5)))

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1500
           
fig.show()

st.plotly_chart(fig)

st.markdown('\n\n')
st.markdown('We tried to find happiness scores that changed over the years to gain some insight. For example:')

############ Thomas' Correlation analysis
fig, axes = plt.subplots(3, 2, figsize=(16,12))
# Graph 1 - Netflix Price vs Explained by: Log GDP per capita
sns.set_context("paper")
tips = df
sns.set_style("whitegrid")
sns.regplot(ax=axes[0][0],x='Netflix Price', y = 'Explained by: Log GDP per capita', data = tips).set_title("Netflix Price vs Explained by: Log GDP per capita", 
                                                                                   fontdict = { 'fontsize': 15})
sns.despine()
axes[0][0].set(xlabel=None)


# Graph 2 - Netflix Price vs Explained by: Social support
sns.set_context("paper")
tips = df
sns.set_style("whitegrid")
sns.regplot(ax=axes[0][1],x='Netflix Price', y = 'Explained by: Social support', data = tips).set_title("Netflix Price vs Explained by: Social support", 
                                                                                   fontdict = { 'fontsize': 15})
sns.despine()
axes[0][1].set(xlabel=None)

# Graph 3 - Netflix Price vs Explained by: Healthy life expectancy
sns.set_context("paper")
tips = df
sns.set_style("whitegrid")
sns.regplot(ax=axes[1][0],x='Netflix Price', y = 'Explained by: Healthy life expectancy', data = tips).set_title("Netflix Price vs Explained by: Healthy life expectancy", 
                                                                                   fontdict = { 'fontsize': 15})
sns.despine()
axes[1][0].set(xlabel=None)

# Graph 4 - Netflix Price vs Explained by: Freedom to make life choices
sns.set_context("paper")
tips = df
sns.set_style("whitegrid")
sns.regplot(ax=axes[1][1],x='Netflix Price', y = 'Explained by: Freedom to make life choices', data = tips).set_title("Netflix Price vs Explained by: Freedom to make life choices", 
                                                                                   fontdict = { 'fontsize': 15})
sns.despine()
axes[1][1].set(xlabel=None)

# Graph 5 - Netflix Price vs Explained by: Generosity
sns.set_context("paper")
tips = df
sns.set_style("whitegrid")
sns.regplot(ax=axes[2][0],x='Netflix Price', y = 'Explained by: Generosity', data = tips).set_title("Netflix Price vs Explained by: Generosity", 
                                                                                   fontdict = { 'fontsize': 15})
sns.despine()

# Graph 6 - Netflix Price vs Explained by: Perceptions of corruption
sns.set_context("paper")
tips = df
sns.set_style("whitegrid")
sns.regplot(ax=axes[2][1],x='Netflix Price', y = 'Explained by: Perceptions of corruption', data = tips).set_title("Netflix Price vs Explained by: Perceptions of corruption", 
                                                                                   fontdict = { 'fontsize': 15})
sns.despine()


# Set x,y-axis label

axes[2][0].set_xlabel('Netflix Price', size = 12)
axes[2][1].set_xlabel('Netflix Price', size = 12)

# Adjust the subplot layout parameters
fig.subplots_adjust(hspace=0.3, wspace=0.2)
st.pyplot(fig)
#################
def heat_2d_hist(var1, var2, df, density=True):
    
    H, xedges, yedges = np.histogram2d(df[var1], df[var2], density=density)
    H[H == 0] = np.nan

    # Create a nice variable that shows the bin boundaries
    xedges = pd.Series(['{0:.2g}'.format(num) for num in xedges])
    xedges = pd.DataFrame({"a": xedges.shift(), "b": xedges}).dropna().agg(' - '.join, axis=1)
    yedges = pd.Series(['{0:.2g}'.format(num) for num in yedges])
    yedges = pd.DataFrame({"a": yedges.shift(), "b": yedges}).dropna().agg(' - '.join, axis=1)

    # from wide to narrow format using melt
    res = pd.DataFrame(H, 
                       index=yedges, 
                       columns=xedges).reset_index().melt(
                            id_vars='index'
                       ).rename(columns={'index': 'feature2', 
                                         'value': 'proportion',
                                         'variable': 'feature'})
    

    # raw left boundary of the bin added as a column, will be used to sort axis labels
    res['raw_left_value'] = res['feature'].str.split(' - ').map(lambda x: x[0]).astype(float)   
    res['raw_left_value2'] = res['feature2'].str.split(' - ').map(lambda x: x[0]).astype(float) 
    res['variable'] = var1
    res['variable2'] = var2 
    
    return res.dropna() # Drop all combinations for which no values where found


################# Interactive Heatmap of Correlations

Cor_df = df

Cor_df.dropna(inplace=True)
## Drop those label columns
value_columns = Cor_df.columns.drop(['Country', 'Region'])
Netflix_data_2dbinned = pd.concat([heat_2d_hist(var1, var2, Cor_df) for var1 in value_columns for var2 in value_columns])
Netflix_data_2dbinned.head()


####### Correlation Data
cor_data = (df.drop(columns=['Country'])
              .corr().stack()
              .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
              .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
cor_data.head()


Netflix_data_2dbinned.query('variable == "Netflix Price"')['variable2'].unique()
Price_vs_Happiness = Netflix_data_2dbinned.query('(variable == "Netflix Price") & (variable2 == "Happiness score")')
feature_sel_cor = alt.selection_single(fields=['variable', 'variable2'], clear=False, 
                                  init={'variable': 'Inflation', 'variable2': 'Generosity'})

# creation of the correlation heatmap
base = alt.Chart(cor_data).encode(
        x='variable:O',
        y='variable2:O', 
    
        )

text = base.mark_text().encode(
        text='correlation_label',
        color=alt.condition(
              alt.datum.correlation > 0.5, 
              alt.value('white'),
              alt.value('black')
              )
        )

cor_plot = base.mark_rect().encode(
           color=alt.condition(feature_sel_cor, alt.value('pink'), 'correlation:Q', scale=alt.Scale(range= ['#336666', '#FFFFFF', '#336666']))
           ).add_selection(feature_sel_cor)

# Creation of the 2d binned histogram plot
scat_plot = alt.Chart(Netflix_data_2dbinned).transform_filter(
            feature_sel_cor
            ).mark_rect().encode(
            alt.X('feature:N', sort=alt.EncodingSortField(field='raw_left_value')), 
            alt.Y('feature2:N', sort=alt.EncodingSortField(field='raw_left_value2', order='descending')),
            alt.Color('proportion:Q', scale=alt.Scale(range= ['#336666', '#FFFFFF', '#91b8bd']))
            )
#scheme='greens'
#color = alt.Color('Happiness score', scale=alt.Scale(range= ['#336666', '#acc8d4', '#91b8bd']
            #strokeDash = 'Country_Name'
#            ))
# Combine both plots. hconcat plots, put both side-by-side with titles
final = alt.hconcat((cor_plot + text).properties(width=500, height=500,
            title = {
                    'text': 'Correlation between features',
                    'subtitle': '(Correlation spotted between Netflix price with GDP, Life Expectancy, Happiness Score)',
                    'subtitleColor': 'grey',
                    'subtitleFontSize': 13
                    #'subtitle': (excluding high income countries
                    # 'subtitleColor': 'grey'
                    # 'subtitleFontSize': 20
                }
        
        ), scat_plot.properties(width=350, height=350,
             title = {
                    'text': 'Distribution of data points',
                     
                     }          
                    )).resolve_scale(color='independent')

## Faced a error parsing the figures, googled on the error, fix as follows
alt.data_transformers.disable_max_rows()

st.altair_chart(final)


# ######################################################################

# st.pyplot(fig)

st.info(
    "Further analysis is needed for project to be robust.")