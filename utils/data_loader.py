import pandas as pd
import streamlit as st


class DataLoader:
    """Utility class handling file imports across multiple formats."""

    @staticmethod
    def parse_file(uploaded_file) -> pd.DataFrame:
        filename = uploaded_file.name
        try:
            if filename.endswith(".csv"):
                return pd.read_csv(uploaded_file)
            elif filename.endswith((".xls", ".xlsx")):
                return pd.read_excel(uploaded_file)
            elif filename.endswith(".txt"):
                return pd.read_csv(uploaded_file, sep=None, engine="python")
            else:
                raise ValueError("Format not supported. Use .csv, .xls, .xlsx, or .txt")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return pd.DataFrame()
