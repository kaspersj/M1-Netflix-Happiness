import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

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


x = np.arange(100)
source = pd.DataFrame({
  'x': x,
  'f(x)': np.sin(x / 5)
})

example_chart = alt.Chart(source).mark_line().encode(
    x='x',
    y='f(x)'
)

st.dataframe(source)

st.altair_chart(example_chart)



