import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.title('Our Netflix Analysis')

x = np.arange(100)
source = pd.DataFrame({
  'x': x,
  'f(x)': np.sin(x / 5)
})

example_chart = alt.Chart(source).mark_line().encode(
    x='x',
    y='f(x)'
)

st.DataFrame(source)

st.altair_chart(example_chart)