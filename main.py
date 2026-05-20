import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import io

from ui import apply_premium_styles
from core import ODESolver
from utils import DataLoader, build_cinematic_plot
import plotly.graph_objects as go

# Configurare Pagină și UI
st.set_page_config(page_title="ODE Numerical Solver", layout="wide", page_icon="🌌")
apply_premium_styles()

st.title("✨ Project: Advanced Numerical Methods Analyzer")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem;'>Interactive resolution of ODEs (First-Order) mapping mathematical abstractions onto computed reality.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration Hub")

    input_mode = st.radio(
        "Data Source:",
        ["Default Scenario (Demo)", "Manual Input", "Upload File (.txt, .xls, .csv)"],
    )
    st.divider()

    func_str, exact_func_str = "", ""
    x0, y0, x_end, steps = 0.0, 1.0, 2.0, 10

    if input_mode == "Default Scenario (Demo)":
        st.success("Demo logic applied automatically.")
        func_str = "x - y + 2"
        exact_func_str = "x + 1 + exp(-x)"
        x0, y0, x_end, steps = 0.0, 2.0, 3.0, 15

        st.markdown("<br>**Differential Equation:**", unsafe_allow_html=True)
        st.latex(r"\frac{dy}{dx} = " + func_str)
        st.markdown("**Analytical (Exact) Solution:**")
        st.latex(r"y(x) = " + exact_func_str)

    elif input_mode == "Manual Input":
        func_str = st.text_input("Equation y' = f(x, y)", value="x + y")
        exact_func_str = st.text_input("Exact Solution y(x) (optional)", value="")
        col1, col2 = st.columns(2)
        x0 = col1.number_input("Initial x0", value=0.0)
        y0 = col2.number_input("Initial y0", value=1.0)
        col3, col4 = st.columns(2)
        x_end = col3.number_input("Target x_end", value=2.0)
        steps = col4.number_input("Steps (N)", min_value=1, value=10)

    else:
        uploaded_file = st.file_uploader(
            "Import Dataset", type=["csv", "xls", "xlsx", "txt"]
        )
        st.caption("Expected columns: func, x0, y0, x_end, steps, exact_func")
        if uploaded_file:
            df_input = DataLoader.parse_file(uploaded_file)
            if not df_input.empty:
                func_str = str(df_input["func"].iloc[0])
                x0 = float(df_input["x0"].iloc[0])
                y0 = float(df_input["y0"].iloc[0])
                x_end = float(df_input["x_end"].iloc[0])
                steps = int(df_input["steps"].iloc[0])
                exact_func_str = (
                    str(df_input["exact_func"].iloc[0])
                    if "exact_func" in df_input.columns
                    else ""
                )
                st.success(f"Parsed file: {uploaded_file.name}")
            else:
                st.stop()
        else:
            st.warning("Awaiting file upload...")
            st.stop()

    st.divider()
    st.subheader("👁️ Visual Layers")
    show_euler = st.checkbox("Euler Method", value=True)
    show_heun = st.checkbox("Heun Method", value=True)
    show_rk4 = st.checkbox("Runge-Kutta 4 (RK4)", value=True)
    show_exact = st.checkbox("Exact Solution", value=True)

    st.markdown("💡 **Pro Features**")
    show_field = st.checkbox("Display Slope Vector Field", value=True)

    selected_methods = [
        m
        for m, show in zip(
            ["Euler", "Heun", "RK4", "Exact"],
            [show_euler, show_heun, show_rk4, show_exact],
        )
        if show
    ]

# Procesarea Datelor și Randarea
if st.button("🚀 Execute Analysis", type="primary", use_container_width=True):
    try:
        solver = ODESolver(func_str, x0, y0, x_end, steps)
        x_vals = solver.get_x_grid()

        results = {}
        if "Euler" in selected_methods:
            results["Euler"] = solver.solve_euler()
        if "Heun" in selected_methods:
            results["Heun"] = solver.solve_heun()
        if "RK4" in selected_methods:
            results["RK4"] = solver.solve_rk4()

        has_exact = (
            exact_func_str
            and exact_func_str.lower() != "nan"
            and "Exact" in selected_methods
        )
        if has_exact:
            x_sym = sp.symbols("x")
            exact_f = sp.lambdify(x_sym, sp.sympify(exact_func_str), "numpy")
            results["Exact"] = exact_f(x_vals)

        df_results = pd.DataFrame({"x": x_vals})
        for method, y_vals in results.items():
            df_results[method] = y_vals

        if has_exact:
            for method in ["Euler", "Heun", "RK4"]:
                if method in results:
                    df_results[f"Error_{method}"] = np.abs(
                        results["Exact"] - results[method]
                    )

        tab1, tab2, tab3 = st.tabs(
            ["🎥 Visual Animation", "📊 Data Matrix", "📉 Error Analysis"]
        )

        with tab1:
            if not selected_methods:
                st.warning("Enable at least one method from the sidebar.")
            else:
                fig = build_cinematic_plot(
                    x_vals, results, selected_methods, solver.f, show_field
                )
                st.plotly_chart(fig, theme=None, use_container_width=True)

        with tab2:
            st.dataframe(df_results.style.format("{:.6f}"), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Export CSV",
                    data=df_results.to_csv(index=False).encode("utf-8"),
                    file_name="ode.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_results.to_excel(writer, index=False)
                st.download_button(
                    "📥 Export Excel",
                    data=output.getvalue(),
                    file_name="ode.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True,
                )

        with tab3:
            if has_exact:
                fig_err = go.Figure()
                colors = {
                    "Error_Euler": "#f43f5e",
                    "Error_Heun": "#3b82f6",
                    "Error_RK4": "#10b981",
                }
                for method in ["Euler", "Heun", "RK4"]:
                    if f"Error_{method}" in df_results.columns:
                        fig_err.add_trace(
                            go.Scatter(
                                x=x_vals[1:],
                                y=df_results[f"Error_{method}"][1:],
                                mode="lines+markers",
                                name=f"{method} Error",
                                line=dict(color=colors.get(f"Error_{method}", "#fff")),
                            )
                        )

                fig_err.update_layout(
                    yaxis_type="log",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    title="Absolute Error Convergence (Log Scale)",
                )
                st.plotly_chart(fig_err, theme=None, use_container_width=True)
            else:
                st.warning("Exact solution is required for error analysis.")

    except Exception as e:
        st.error(f"Computation Error: {str(e)}")
