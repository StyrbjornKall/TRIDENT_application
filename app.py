import streamlit as st
from pages import home_page, trident1_page, trident2_page, documentation_page, publications_page, contact_page, space_page

st.set_page_config(
    layout="wide",
    page_title="TRIDENT - chemical ecotoxicity prediction",
    page_icon=":material/water_drop:",
    menu_items={
        "Get help": "mailto:styrbjorn.kall@gu.se",
        "Report a bug": "mailto:styrbjorn.kall@gu.se",
        "About": (
            "# TRIDENT\n"
            "Deep learning models for predicting chemical ecotoxicity endpoints from chemical "
            "structure: TRIDENT-1 for standard aquatic test species, and TRIDENT-2 for any "
            "species with an NCBI taxonomy identifier.\n\n"
            "Developed at the University of Gothenburg."
        ),
    },
)

# Import shared styles
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

### APP #############################

predict_pg = st.Page(trident1_page.print_trident1_predict_page, title="Use TRIDENT-1", icon=":material/set_meal:")
trident2_pg = st.Page(trident2_page.print_trident2_predict_page, title="Use TRIDENT-2", icon=":material/public:")
publications_pg = st.Page(publications_page.print_publications_page, title="Publications", icon=":material/article:")
docs_pg = st.Page(documentation_page.print_doc_page, title="Documentation", icon=":material/menu_book:")
space_pg = st.Page(space_page.print_space_page, title="Explore the chemical space of TRIDENT-1", icon=":material/scatter_plot:")
contact_pg = st.Page(contact_page.print_contact_page, title="Contact", icon=":material/mail:")
home_pg = st.Page(
    lambda: home_page.print_home_page(predict_pg, trident2_pg, docs_pg),
    title="Home",
    icon=":material/home:",
    default=True,
)

pg = st.navigation(
    [home_pg, predict_pg, trident2_pg, publications_pg, docs_pg, space_pg, contact_pg],
    position="top",
)

pg.run()

with st.bottom:
    st.markdown(
        '<div class="app-footer">'
        '<span>© 2026 Styrbjörn Käll. All rights reserved.</span>'
        '</div>',
        unsafe_allow_html=True,
    )