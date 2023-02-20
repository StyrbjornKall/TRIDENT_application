import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def print_contact_page():
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title('Contact')
        st.subheader('Send us an email')
        st.markdown('Please send us an email to report bugs, suggest improvements or if encountering SMILES yielding erroneous outputs.')
        with st.form("my_form", clear_on_submit=True):
            name = st.text_input('Name')
            affiliation = st.text_input('Affiliation (Optional)')
            email = st.text_input('email')
            message = st.text_area('Message')

            upload = st.file_uploader('Attach file')

            # Every form must have a submit button.
            submitted = st.form_submit_button("Send")

            if submitted:
                __send_email(name, affiliation, email, message, upload)
                st.success("Your message has been sent.")


# Define email function
def __send_email(name, affiliation, email, message, upload):
    
    # Set up message
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = 'ecocaithelpdesk@gmail.com'
    msg['Subject'] = 'TICKET: From ecoCAIT.streamlit.app contact form'
    msg.attach(MIMEText(message, 'plain'))

    # Set up SMTP server and send message
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.sendmail(email, 'ecocaithelpdesk@streamlit.com', msg.as_string())
    server.quit()