import pandas as pd
import streamlit as st


class DataLoader:
    """Helper to load input data from various file types."""

    @staticmethod
    @st.cache_data(
        show_spinner=False
    )  # cache to prevent reloading the file on every UI interaction
    def parse_file(uploaded_file) -> pd.DataFrame:
        filename = uploaded_file.name
        try:
            if filename.endswith(".csv"):
                return pd.read_csv(uploaded_file)
            elif filename.endswith((".xls", ".xlsx")):
                return pd.read_excel(uploaded_file)
            elif filename.endswith(".txt"):
                # handles basic text files with common delimiters
                return pd.read_csv(uploaded_file, sep=None, engine="python")
            else:
                raise ValueError(
                    "Unsupported format. Please use .csv, .xls, .xlsx, or .txt"
                )
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return pd.DataFrame()
