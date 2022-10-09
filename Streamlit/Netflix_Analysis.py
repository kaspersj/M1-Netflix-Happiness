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

netflix_pricing_df = pd.read_csv('Netflix_Pricing.csv')

# Happiness Dataframe #
happiness_df = pd.read_csv('world-happiness-report-2021.csv')
happiness_df = happiness_df.drop(['Dystopia + residual', 'lowerwhisker', 'upperwhisker'], axis = 1)
happiness_df = happiness_df.rename(columns={'Country name': 'Country'})
happiness_df = happiness_df.replace(to_replace="Taiwan Province of China", value = "Taiwan")
happiness_df = happiness_df.replace(to_replace="Bosnia and Herzegovina", value = "Bosnia & Herzegovina")
happiness_df = happiness_df.replace(to_replace="Hong Kong S.A.R. of China", value = "Hong Kong")
happiness_df = happiness_df.replace(to_replace="Palestinian Territories", value = "Palestine")
happiness_df = happiness_df.replace(to_replace="Russia", value = "Russian Federation")

# Visualizing the happiness_df #
aly.dist(happiness_df)
aly.dist(happiness_df, mark='bar')
aly.pair(happiness_df)

# CPI Dataframe #
CPI_Inflation_df = pd.read_csv('CPI_Inflation.csv')
CPI_Inflation_df_1 = CPI_Inflation_df.loc[:, ["Country Name","2020"]]
CPI_Inflation_df_1 = CPI_Inflation_df_1.rename(columns={'2020': 'Inflation'})

# Population Dataframe #
Population_df = pd.read_excel('Pop_Data.xls')
Population_df_1 = Population_df.loc[:, ["Country Name","2020"]]
Population_df_1 = Population_df_1.rename(columns={'2020': 'Population'})

# Indicators Dataframe #
indicators_df=pd.merge(CPI_Inflation_df_1,Population_df_1,on='Country Name')
indicators_df = indicators_df.rename(columns={'Country Name': 'Country'})
indicators_df = indicators_df.replace(to_replace="Slovak Republic", value = "Slovakia")
indicators_df = indicators_df.replace(to_replace="Korea, Rep.", value = "South Korea")
indicators_df = indicators_df.replace(to_replace="Turkiye", value = "Turkey")
indicators_df = indicators_df.replace(to_replace="Venezuela, RB", value = "Venezuela")
indicators_df = indicators_df.replace(to_replace="Yemen, Rep.", value = "Yemen")

# Netflix Dataframe #
netflix_df_raw = pd.read_csv('Netflix_Pricing.csv')
netflix_df_raw = netflix_df_raw.set_index("Country")
netflix_df2_raw = netflix_df_raw.drop(["Maximum", "Minimum", "Average"]).reset_index()
netflix_df2_raw.drop(netflix_df2_raw.tail(1).index,inplace=True)

##### Filtering Netflix Dataframe ####
features = ['Country', 'Price USD', '# of TV Shows', '# of Movies', 'Total Library Size', 'Price per Title']

def df_filtered(df):
    
    condition_1 = df.columns.isin(features)
    filtered_df_cond = df.columns[condition_1]
    df1=df[filtered_df_cond]
    ## Order the df column labels 
    df1 = df1[["Country", 
               "Price USD", 
               "# of TV Shows", 
               "# of Movies",
               "Total Library Size",
               "Price per Title"              
              ]]
    
    #filtered_df = df1.groupby(['Region']).mean()
    
    return df1
#########################

netflix_df=df_filtered(netflix_df2_raw)

####### Making our main Dataframe ##### 
def main_df():


    simi_merged_df=netflix_df.merge(happiness_df, on='Country', how = 'left')
    df=simi_merged_df.merge(indicators_df, on='Country', how = 'left')

    df = df.rename(columns={'Regional indicator': 'Region',
                            'Price USD':'Netflix Price'})
    first_column = df.pop('Region')
    df.insert(0, 'Region', first_column)

    # Replace Gibraltar
    df.at[37, 'Region']='Western Europe'
    # Replace Bermuda
    df.at[47, 'Region']='North America and ANZ'
    # Replace Qatar
    df.at[54, 'Region']='Middle East and North Africa'
    # Replace Monaco
    df.at[57, 'Region']='Western Europe'
    # Replace Andorra
    df.at[69, 'Region']='Middle East and North Africa'
    # Replace San Marino
    df.at[74, 'Region']='Western Europe'
    # Replace Oman
    df.at[77, 'Region']='Middle East and North Africa'
    # Replace French Guiana
    df.at[80, 'Region']='Western Europe'
    # Replace French Polynesia
    df.at[83, 'Region']='Western Europe'
    # Replace Liechtenstein
    df.at[87, 'Region']='Western Europe'

    return df
############## Defining df, and filtering ###################

df = main_df()
df = df.rename(columns={'Regional indicator': 'Region',
                       'Price USD':'Netflix Price'})
first_column = df.pop('Region')
df.insert(0, 'Region', first_column)

# Replace Gibraltar
df.at[37, 'Region']='Western Europe'
# Replace Bermuda
df.at[47, 'Region']='North America and ANZ'
# Replace Qatar
df.at[54, 'Region']='Middle East and North Africa'
# Replace Monaco
df.at[57, 'Region']='Western Europe'
# Replace Andorra
df.at[69, 'Region']='Middle East and North Africa'
# Replace San Marino
df.at[74, 'Region']='Western Europe'
# Replace Oman
df.at[77, 'Region']='Middle East and North Africa'
# Replace French Guiana
df.at[80, 'Region']='Western Europe'
# Replace French Polynesia
df.at[83, 'Region']='Western Europe'
# Replace Liechtenstein
df.at[87, 'Region']='Western Europe'

########################### Visualizing df ################################
aly.nan(df)
aly.pair(df, 'Region')

cor_data = (df.drop(columns=['Country'])
              .corr().stack()
              .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
              .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal


############################ Correlation viz  #############################################
base = alt.Chart(cor_data).encode(
    x='variable2:O',
    y='variable:O'    
)

# Text layer with correlation labels
# Colors are for easier readability
text = base.mark_text().encode(
    text='correlation_label',
    color=alt.condition(
        alt.datum.correlation > 0.5, 
        alt.value('white'),
        alt.value('black')
    )
)

# The correlation heatmap itself
cor_plot = base.mark_rect().encode(
    color='correlation:Q'
)

final_chart = (cor_plot + text).properties(
                height = 600,
                width = 600,
                title = {
                    'text': 'Correlation between features',
                    'subtitle': '(Correlation spotted between Netflix price with GDP, Life Expectancy, Happiness Score)',
                    'subtitleColor': 'grey',
                    'subtitleFontSize': 16
                    #'subtitle': (excluding high income countries
                    # 'subtitleColor': 'grey'
                    # 'subtitleFontSize': 20
                }
               
                ).configure_title(
                    anchor = 'start',
                    color = 'black',
                    fontSize = 20,
                    subtitleColor = 'grey',
                    subtitleFontSize = 20
                    )

st.altair_chart(final_chart)

############################  ######################################## 
aly.corr(df)

############################ 2D histogram #############################

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

############################# Cleaning up df ######################################
Cor_df = df

Cor_df.dropna(inplace=True)
## Drop those label columns
value_columns = Cor_df.columns.drop(['Country', 'Region'])
Netflix_data_2dbinned = pd.concat([heat_2d_hist(var1, var2, Cor_df) for var1 in value_columns for var2 in value_columns])


############################## Correlation Heatmap #########################################
Netflix_data_2dbinned.query('variable == "Netflix Price"')['variable2'].unique()
Price_vs_Happiness = Netflix_data_2dbinned.query('(variable == "Netflix Price") & (variable2 == "Ladder score")')
feature_sel_cor = alt.selection_single(fields=['variable', 'variable2'], clear=False, 
                                  init={'variable': 'Inflation', 'variable2': 'Generosity'})

# creation of the correlation heatmap
base = alt.Chart(cor_data).encode(
        x='variable:O',
        y='variable2:O'    
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
           color=alt.condition(feature_sel_cor, alt.value('pink'), 'correlation:Q')
           ).add_selection(feature_sel_cor)

# Creation of the 2d binned histogram plot
scat_plot = alt.Chart(Netflix_data_2dbinned).transform_filter(
            feature_sel_cor
            ).mark_rect().encode(
            alt.X('feature:N', sort=alt.EncodingSortField(field='raw_left_value')), 
            alt.Y('feature2:N', sort=alt.EncodingSortField(field='raw_left_value2', order='descending')),
            alt.Color('proportion:Q', scale=alt.Scale(scheme='blues'))
            )

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



st.altair_chart(final)


############### Global Netflix Pricing Correlation Analysis by Region ##################
fig, ax = plt.subplots(1,4, figsize=(16, 6), sharey=True)

fig.suptitle("Global Vs Western Europe vs North America vs Central and Eastern Europe Pricing Correlation Analysis", fontsize=16)

global_corr = df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)
WE_df = df[df['Region']=='Western Europe']
WE_corr = WE_df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)

NA_df = df[df['Region']=='North America and ANZ']
NA_corr = NA_df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)

CE_df = df[df['Region']=='Central and Eastern Europe']
CE_corr = CE_df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)



sns.heatmap(global_corr, ax=ax[0], annot=True)
sns.heatmap(WE_corr, ax=ax[1], annot=True)
sns.heatmap(NA_corr, ax=ax[2], annot=True)
sns.heatmap(CE_corr, ax=ax[3], annot=True)

ax[0].title.set_text('Global')
ax[1].title.set_text('Western Europe')
ax[2].title.set_text('North America and ANZ')
ax[3].title.set_text('Central and Eastern Europe')


st.pyplot(fig) #hopefullly this plots correctly




##### Another Netflix Pricing correlation analysis by region: ###########
fig, ax = plt.subplots(1,4, figsize=(16, 6), sharey=True)

fig.suptitle("Middle East Vs Latin America vs East Asia vs South East Asia Pricing Correlation Analysis", fontsize=16)

ME_df = df[df['Region']=='Middle East and North Africa']
ME_corr = WE_df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)

LAC_df = df[df['Region']=='Latin America and Caribbean']
LAC_corr = LAC_df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)

EA_df = df[df['Region']=='East Asia']
EA_corr = EA_df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)

SEA_df = df[df['Region']=='Southeast Asia']
SEA_corr = SEA_df.corr()[['Netflix Price']].sort_values(by = 'Netflix Price', ascending = False)

sns.heatmap(ME_corr, ax=ax[0], annot=True)
sns.heatmap(LAC_corr, ax=ax[1], annot=True)
sns.heatmap(EA_corr, ax=ax[2], annot=True)
sns.heatmap(SEA_corr, ax=ax[3], annot=True)


ax[0].title.set_text('Middle East and North Africa')
ax[1].title.set_text('Latin America and Caribbean')
ax[2].title.set_text('East Asia')
ax[3].title.set_text('Southeast Asia')

st.pyplot(fig)


############### Correlation analysis with Southeast Asia ##################
aly.corr(df[df['Region']=='Southeast Asia'])


############## World map of Happiness, Total Library Size, and European countries ##### 
revised_df = df.dropna() # removes na values 

fig = px.scatter_geo(revised_df, locations="Country", locationmode = 'country names',
                     color="Ladder score", # sets the color of markers
                     hover_name="Country", 
                     size="Total Library Size", # size of markers
                     scope = 'europe',
                     title = 'Happiness and Library Size'
)
# Scope can take the following values:
#[‘world’, ‘usa’, ‘europe’, ‘asia’, ‘africa’, ‘north america’, ‘south america’]
# fig.show()

st.pyplot(fig)


############ Logged GDP per capita vs Happiness score by Country ######
df = df.rename(columns={"Ladder score": "Happiness score"})
fig=px.scatter(df,
              x='Happiness score',
              y='Logged GDP per capita',
              size='Netflix Price',
              template='plotly_white',
              color='Region',
              hover_name='Country',
              labels={"Netflix Price": "Size", "Region": "Region"},
              size_max=40)
fig.update_layout(title='Logged GDP per capita vs Happiness score by Country')

st.pyplot(fig)

######################################################################
# Outliers
######################################################################


