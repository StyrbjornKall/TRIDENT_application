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

        contact_form = """
            <style>
            form {
                max-width: 600px;
                margin: 0 auto;
                background-color: #f4f4f4;
                padding: 20px;
                border-radius: 5px;
            }

            .form-group {
                margin-bottom: 20px;
            }

            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }

            input[type="text"],
            input[type="email"],
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 16px;
                box-sizing: border-box;
            }

            input[type="file"] {
                margin-top: 10px;
            }

            button[type="submit"] {
                background-color: #F63366;
                color: #fff;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            
            </style>

            <form enctype="multipart/form-data" action="https://formsubmit.co/kall.styrbjorn@gmail.com" method="POST">
            <div class="form-group">
                <label for="name">Name</label>
                <input type="text" id="name" name="name" placeholder="Enter your name" required>
            </div>
            
            <div class="form-group">
                <label for="issue">Issue description</label>
                <textarea id="issue" name="issue" placeholder="Enter a brief description of your issue" required></textarea>
            </div>

            <div class="form-group">
                <label for="email">Email address</label>
                <input type="email" id="email" name="email" placeholder="Enter your email address">
            </div>

            <div class="form-group">
                <label for="attachment">Attachment (<5 MB)</label>
                <input type="file" id="attachment" name="attachment" accept=".png, .jpeg, .csv, .txt">
            </div>

            <input type="hidden" name="_subject" value="Ticket from TRIDENT web app">
            <input type="hidden" name="_autoresponse" value="We have recieved your form. Please don't hesitate to contact us through tridenthelpdesk@gmail.com if you have any further questions.">
            <input type="hidden" name="_template" value="table">
            
            <div class="form-group">
                <button type="submit">Send</button>
            </div>
            </form>
            """
        st.markdown(contact_form, unsafe_allow_html=True)
    
    author_text = '''<div style="display: flex; justify-content: space-between; padding-top: 60px">
    <div style="flex-basis: 33.33%; text-align: center;">
        <h3>Mikael Gustavsson</h3>
        <p><small>Department of Economics, University of Gothenburg<br>Gothenburg, Sweden<br>
        mikael.gustavsson@chalmers.se</small></p>
    </div>
    <div style="flex-basis: 33.33%; text-align: center;">
        <h3>Styrbjörn Käll</h3>
        <p><small>Department of Mathematical Sciences, Chalmers University of Technology<br>Gothenburg, Sweden<br>
        kall.styrbjorn@gmail.com</small></p>
    </div>
    <div style="flex-basis: 33.33%; text-align: center;">
        <h3>Erik Kristiansson</h3>
        <p><small>Department of Mathematical Sciences, Chalmers University of Technology<br>Gothenburg, Sweden<br>
        erik.kristiansson@chalmers.se</small></p>
    </div>
</div>
'''

    st.markdown(author_text, unsafe_allow_html=True)