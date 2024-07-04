import streamlit as st
from streamlit_option_menu import option_menu
from validation import validation_page
from database import database_page
from history import history_page


st.subheader('Cam-Cord :camera_with_flash:')

st.sidebar.subheader('About')
st.sidebar.info('The Cam-Cord project is a face recognition system designed for tracking visitor history.'
                'It involves capturing and storing video footage of visitors, processing the data for facial recognition,'
                'and maintaining a detailed log of visitors. The project is implemented using Python and OpenCV,'
                'ensuring real-time detection and efficient data handling.')
st.sidebar.divider()
clr = st.sidebar.button('Clear Data')
st.sidebar.warning('Please check before clearing data.')


selected_menu = option_menu(None,
                            ['Visitors', 'Add to DB', 'View History'],
                            icons=['camera', 'person-plus', 'clock-history'],
                            menu_icon="cast", default_index=0, orientation="horizontal")


if selected_menu == 'Visitors':
    validation_page()

elif selected_menu == 'Add to DB':
    database_page()

elif selected_menu == 'View History':
    history_page()
