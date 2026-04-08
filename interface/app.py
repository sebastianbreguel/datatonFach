import streamlit as st
from views.prediction import show_prediction
from views.quantification import show_quantification


def main():
    st.set_page_config(page_title="Analisis de incendios", layout="wide")
    st.sidebar.title("Analisis de incendios")
    pages = {"Cuantificador de incendios 🔢": show_quantification, "Predictor de incendios ⏩️": show_prediction}

    # Use query params for navigation to retain state across reloads
    page = "Cuantificador de incendios 🔢"

    # Radio buttons for navigation
    page_selection = st.sidebar.radio("Ir a", list(pages.keys()), index=list(pages.keys()).index(page))

    # Update query params when the page changes
    if page_selection != page:
        st.query_params["page"] = list(pages.keys()).index(page_selection)
        page = page_selection

    # Display the selected page
    pages[page]()


if __name__ == "__main__":
    main()
