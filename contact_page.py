import streamlit as st

def print_contact_page():
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title('Contact')
        st.subheader('Send us an email')
        with st.form("my_form", clear_on_submit=True):
            name = st.text_input('Name')
            email = st.text_input('email')
            message = st.text_area('Message')

            upload = st.file_uploader('Attach file')

            # Every form must have a submit button.
            submitted = st.form_submit_button("Send")
