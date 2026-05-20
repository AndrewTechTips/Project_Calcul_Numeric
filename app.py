import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import io

from ui import apply_premium_styles
from core import ODESolver
from utils import DataLoader, build_cinematic_plot
import plotly.graph_objects as go

# Configurare Pagină și UI de înaltă rezoluție
st.set_page_config(
    page_title="ODE Numerical Solver", layout="wide", page_icon=":material/functions:"
)
apply_premium_styles()

st.title(":material/science: Project: Advanced Numerical Methods Analyzer")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem;'>Interactive resolution of ODEs (First-Order) mapping mathematical abstractions onto computed reality.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(":material/settings: Configuration Hub")

    input_mode = st.radio(
        "Input Paradigm:",
        [
            "Preset Mathematical Models",
            "Custom Manual Input",
            "Upload Dataset (.txt, .xls, .csv)",
        ],
    )
    st.divider()

    # Inițializare variabile globale pentru controlul stării widget-urilor
    func_str, exact_func_str = "x + y", ""
    x0, y0, x_end, steps = 0.0, 1.0, 2.0, 10

    if input_mode == "Preset Mathematical Models":
        st.markdown("### :material/bookmarks: Select a Template")
        preset_choice = st.selectbox(
            "Choose a standard scientific model:",
            [
                "Logistic Population Growth Model",
                "Newton's Law of Cooling",
                "Pharmacokinetic Model (Continuous)",
                "Exponential Decay Model",
            ],
        )

        # Alinierea cu datele exacte din modelele de laborator
        if preset_choice == "Logistic Population Growth Model":
            st.info(
                "Loaded standard ecosystem carrying-capacity model.",
                icon=":material/groups:",
            )
            func_str = "1 * y * (1 - y / 10)"
            exact_func_str = "10 / (1 + 4 * exp(-x))"
            x0, y0, x_end, steps = 0.0, 2.0, 5.0, 20

        elif preset_choice == "Newton's Law of Cooling":
            st.info(
                "Loaded thermodynamic ambient cooling model.",
                icon=":material/device_thermostat:",
            )
            func_str = "-0.5 * (y - 20)"
            exact_func_str = "20 + 70 * exp(-0.5 * x)"
            x0, y0, x_end, steps = 0.0, 90.0, 5.0, 20

        elif preset_choice == "Pharmacokinetic Model (Continuous)":
            st.info(
                "Loaded drug concentration model with steady-state injection.",
                icon=":material/vaccines:",
            )
            func_str = "-0.4 * y + 8"
            exact_func_str = "20 - 20 * exp(-0.4 * x)"
            x0, y0, x_end, steps = 0.0, 0.0, 5.0, 20

        elif preset_choice == "Exponential Decay Model":
            st.info(
                "Loaded metabolic substance elimination model.",
                icon=":material/hourglass_empty:",
            )
            func_str = "-0.4 * y"
            exact_func_str = "100 * exp(-0.4 * x)"
            x0, y0, x_end, steps = 0.0, 100.0, 5.0, 20

    elif input_mode == "Custom Manual Input":
        st.markdown("### :material/edit_note: Operator Sandbox")
        func_str = st.text_input(
            "First Derivative Equation Expression: y' = f(x, y)", value="x + y"
        )
        exact_func_str = st.text_input(
            "Analytical Exact Solution: y(x) (Optional)", value=""
        )

        # Organizare pe coloane simetrice pentru parametrii de iterație
        with st.container():
            col1, col2 = st.columns(2)
            x0 = col1.number_input("Boundary Left (x0)", value=0.0, format="%.4f")
            y0 = col2.number_input("Initial Condition (y0)", value=1.0, format="%.4f")

            col3, col4 = st.columns(2)
            x_end = col3.number_input(
                "Boundary Right (x_end)", value=2.0, format="%.4f"
            )
            steps = col4.number_input(
                "Mesh Intervals (Steps N)", min_value=1, value=10, step=1
            )

    else:
        st.markdown("### :material/file_upload: Data File Dispatcher")
        uploaded_file = st.file_uploader(
            "Upload configuration dataset", type=["csv", "xls", "xlsx", "txt"]
        )
        st.caption("Schema constraint: [func, x0, y0, x_end, steps, exact_func]")

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
                    and pd.notna(df_input["exact_func"].iloc[0])
                    else ""
                )
                st.toast(
                    f"Configuration loaded: {uploaded_file.name}",
                    icon=":material/check_circle:",
                )
            else:
                st.stop()
        else:
            st.warning(
                "Awaiting secure file stream target...", icon=":material/upload_file:"
            )
            st.stop()

    st.divider()
    st.subheader(":material/layers: Vector Layers")
    show_euler = st.checkbox("Euler Approximation Curve", value=True)
    show_heun = st.checkbox("Heun (Improved Euler) Curve", value=True)
    show_rk4 = st.checkbox("Runge-Kutta Order 4 (RK4) Curve", value=True)
    show_exact = st.checkbox("Analytical Ground Truth Curve", value=True)

    st.markdown("🔧 **System Visualization Pro**")
    show_field = st.checkbox("Display Gradient Slope Field", value=False)

    selected_methods = [
        m
        for m, show in zip(
            ["Euler", "Heun", "RK4", "Exact"],
            [show_euler, show_heun, show_rk4, show_exact],
        )
        if show
    ]

# ==========================================================
# Zona de Visual Preview Premium (Live Mathematical Feedback)
# ==========================================================
st.markdown("### :material/visibility: Live Equation Preview Engine")
preview_container = st.container(border=True)
with preview_container:
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        try:
            parsed_f = sp.sympify(func_str)
            st.markdown(
                "<p style='color: #6366f1; margin-bottom: 2px; font-weight: 600;'>System ODE Representation:</p>",
                unsafe_allow_html=True,
            )
            st.latex(r"\frac{dy}{dx} = " + sp.latex(parsed_f))
        except Exception:
            st.caption(
                ":material:error: Parsing error or incomplete expression in input field."
            )

    with col_pre2:
        if exact_func_str and exact_func_str.lower() != "nan":
            try:
                parsed_exact = sp.sympify(exact_func_str)
                st.markdown(
                    "<p style='color: #10b981; margin-bottom: 2px; font-weight: 600;'>Validated Analytical Solution:</p>",
                    unsafe_allow_html=True,
                )
                st.latex(r"y(x) = " + sp.latex(parsed_exact))
            except Exception:
                st.caption(
                    ":material:question_mark: Optional analytical trace template unparsed."
                )
        else:
            st.markdown(
                "<p style='color: #94a3b8; margin-bottom: 2px; font-weight: 600;'>Validated Analytical Solution:</p>",
                unsafe_allow_html=True,
            )
            st.caption(
                ":material:lock: No continuous function provided. System error checking restricted to relative delta tracking."
            )

st.markdown("<br>", unsafe_allow_html=True)

# Coordonarea execuției algoritmice
if st.button(
    "Run Simulation",
    type="primary",
    use_container_width=True,
    icon=":material/rocket_launch:",
):
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
            [
                ":material/movie: Step-by-Step Evolution",
                ":material/table_chart: Discrete Matrix Report",
                ":material/show_chart: Spectral Convergence Analysis",
            ]
        )

        with tab1:
            if not selected_methods:
                st.warning(
                    "Enable at least one analytical or approximation layer.",
                    icon=":material/warning:",
                )
            else:
                fig = build_cinematic_plot(
                    x_vals, results, selected_methods, solver.f, show_field
                )
                st.plotly_chart(fig, theme=None, use_container_width=True)

        with tab2:
            st.dataframe(df_results.style.format("{:.6f}"), use_container_width=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Export Clean CSV",
                    data=df_results.to_csv(index=False).encode("utf-8"),
                    file_name="ode_output.csv",
                    mime="text/csv",
                    use_container_width=True,
                    icon=":material/download:",
                )
            with col_d2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_results.to_excel(writer, index=False)
                st.download_button(
                    "Export Clean Excel",
                    data=output.getvalue(),
                    file_name="ode_output.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True,
                    icon=":material/table:",
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
                                name=f"{method} Residual",
                                line=dict(
                                    color=colors.get(f"Error_{method}", "#fff"), width=2
                                ),
                            )
                        )

                fig_err.update_layout(
                    yaxis_type="log",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    title="Truncation Error Attenuation Matrix (Logarithmic Scale)",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Domain Interval (X)",
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Log Magnitude |E_n|",
                    ),
                )
                st.plotly_chart(fig_err, theme=None, use_container_width=True)
            else:
                st.warning(
                    "Convergence tracing requires an active analytical evaluation string.",
                    icon=":material/error:",
                )

    except Exception as e:
        st.error(f"Numerical Engine Fault: {str(e)}", icon=":material/bug_report:")
