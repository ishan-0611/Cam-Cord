import streamlit as st
import pandas as pd
import os


def history_page():
    visitors_db_file = 'visitor_database/visitors_db.csv'

    if os.path.exists(visitors_db_file):
        df_visitors = pd.read_csv(visitors_db_file)
        st.subheader("Visitor History")
        st.dataframe(df_visitors)
    else:
        st.error("Visitors database not found.")
