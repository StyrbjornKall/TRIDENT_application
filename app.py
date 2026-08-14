import streamlit as st
import os
import streamlit_analytics2 as streamlit_analytics
from pages import home_page, predict_page, trident2_page, documentation_page, publications_page, contact_page, space_page

st.set_page_config(layout="wide", page_title='TRIDENT - predicting chemical ecotoxicity using AI', page_icon="🔱")

#import styles
primary_color = st.get_option("theme.primaryColor") or "#F63366"
with open('styles.css')as f:
    css = f.read().replace("__PRIMARY_COLOR__", primary_color)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html = True)

### APP #############################

pg = st.navigation(
    [
        st.Page(home_page.print_home_page, title="Home", icon="🔱", default=True),
        st.Page(predict_page.print_predict_page, title="Use TRIDENT-1", icon="🐟"),
        st.Page(trident2_page.print_trident2_page, title="Use TRIDENT-2", icon="🌍"),
        st.Page(publications_page.print_publications_page, title="Publications", icon="📰"),
        st.Page(documentation_page.print_doc_page, title="Documentation", icon="📄"),
        st.Page(space_page.print_space_page, title="Explore the chemical space of TRIDENT-1", icon="🌌"),
        st.Page(contact_page.print_contact_page, title="Contact", icon="✉️"),
    ],
    position="top",
)

with streamlit_analytics.track(
    unsafe_password=os.environ.get("ANALYTICS_PASSWORD", "tridentanalytics"),
    save_to_json="data/analytics_data.json",
    load_from_json="data/analytics_data.json",
):
    pg.run()

with st.bottom:
    st.caption("© 2026 Styrbjörn Käll · All rights reserved", text_alignment="right")