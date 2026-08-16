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
                background-color: #FFFFFF;
                padding: 24px;
                border-radius: 0.4rem;
                border: 1px solid var(--trident-border, #D8DEE9);
                box-shadow: none;
            }

            .form-group {
                margin-bottom: 20px;
            }

            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
            }

            input[type="text"],
            input[type="email"],
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #d8dee9;
                border-radius: 0.35rem;
                font-size: 16px;
                font-family: inherit;
                box-sizing: border-box;
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
            }

            input[type="text"]:focus,
            input[type="email"]:focus,
            textarea:focus {
                outline: none;
                border-color: var(--trident-primary, #5E81AC);
                box-shadow: 0 0 0 3px color-mix(in srgb, var(--trident-primary, #5E81AC) 15%, transparent);
            }

            input[type="file"] {
                margin-top: 10px;
            }

            button[type="submit"] {
                background-color: var(--trident-primary, #5E81AC);
                color: #fff;
                border: none;
                border-radius: 0.35rem;
                padding: 10px 22px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
            }

            button[type="submit"]:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 10px rgba(16, 24, 40, 0.15);
                opacity: 0.95;
            }
            
            </style>

            <form enctype="multipart/form-data" action="https://formsubmit.co/bccb49e3b706b5f42a3b64f6fb55a437" method="POST">
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
            <input type="hidden" name="_autoresponse" value="We have recieved your form. Please don't hesitate to contact us if you have any further questions.">
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
        mikael.gustavsson@bioenv.gu.se</small></p>
    </div>
    <div style="flex-basis: 33.33%; text-align: center;">
        <h3>Styrbjörn Käll</h3>
        <p><small>Knightec Group | Department of Mathematical Sciences, Chalmers University of Technology<br>Gothenburg, Sweden<br>
        styrbjorn.kall@semcon.com</small></p>
    </div>
    <div style="flex-basis: 33.33%; text-align: center;">
        <h3>Erik Kristiansson</h3>
        <p><small>Department of Mathematical Sciences, Chalmers University of Technology<br>Gothenburg, Sweden<br>
        erik.kristiansson@chalmers.se</small></p>
    </div>
</div>
'''

    st.markdown(author_text, unsafe_allow_html=True)
