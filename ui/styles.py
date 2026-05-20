import streamlit as st


def apply_premium_styles():
    st.markdown(
        """
    <style>
        /* Smooth Animated Premium Gradient Background */
        .stApp {
            background: linear-gradient(-45deg, #090e17, #161224, #0b1727, #0d131f);
            background-size: 400% 400%;
            animation: gradientFlow 20s ease infinite;
            color: #f8fafc;
        }
        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Ultra-Glassmorphism for Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(10, 15, 25, 0.55) !important;
            backdrop-filter: blur(28px) !important;
            -webkit-backdrop-filter: blur(28px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 5px 0 30px rgba(0,0,0,0.3);
        }
        header[data-testid="stHeader"] { background: transparent !important; }

        /* Premium Input Elements */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
            transition: all 0.3s ease;
        }
        div[data-baseweb="input"] > div:focus-within {
            border-color: rgba(99, 102, 241, 0.6) !important;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
            background: rgba(255, 255, 255, 0.05) !important;
        }

        /* Tabs and Expanders */
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.015);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
        }

        /* Primary Buttons */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.35);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(79, 70, 229, 0.5);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
