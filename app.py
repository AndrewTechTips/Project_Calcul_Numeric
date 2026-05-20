import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import io

from ui import apply_premium_styles
from core import ODESolver
from utils import DataLoader, build_cinematic_plot
import plotly.graph_objects as go

# setup main page config
st.set_page_config(
    page_title="ODE Solver", layout="wide", page_icon=":material/functions:"
)
apply_premium_styles()

st.title(":material/science: Numerical Methods: ODE Solver")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem;'>A visual solver for first-order ODEs using step-by-step approximation methods.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(":material/settings: Settings")

    input_mode = st.radio(
        "Input Method:",
        ["Templates", "Manual Input", "File Upload (.txt, .xls, .csv)"],
    )
    st.divider()

    # default fallback values
    func_str, exact_func_str = "x + y", ""
    x0, y0, x_end, steps = 0.0, 1.0, 2.0, 10

    if input_mode == "Templates":
        st.markdown("### :material/bookmarks: Select a Template")
        preset_choice = st.selectbox(
            "Choose a predefined model:",
            [
                "Logistic Population Growth",
                "Newton's Law of Cooling",
                "Pharmacokinetic Model",
                "Exponential Decay",
            ],
        )

        # load specific data based on the chosen template
        if preset_choice == "Logistic Population Growth":
            st.info("Logistic model loaded.", icon=":material/groups:")
            func_str = "1 * y * (1 - y / 10)"
            exact_func_str = "10 / (1 + 4 * exp(-x))"
            x0, y0, x_end, steps = 0.0, 2.0, 5.0, 20

        elif preset_choice == "Newton's Law of Cooling":
            st.info("Cooling model loaded.", icon=":material/device_thermostat:")
            func_str = "-0.5 * (y - 20)"
            exact_func_str = "20 + 70 * exp(-0.5 * x)"
            x0, y0, x_end, steps = 0.0, 90.0, 5.0, 20

        elif preset_choice == "Pharmacokinetic Model":
            st.info("Drug concentration model loaded.", icon=":material/vaccines:")
            func_str = "-0.4 * y + 8"
            exact_func_str = "20 - 20 * exp(-0.4 * x)"
            x0, y0, x_end, steps = 0.0, 0.0, 5.0, 20

        elif preset_choice == "Exponential Decay":
            st.info("Decay model loaded.", icon=":material/hourglass_empty:")
            func_str = "-0.4 * y"
            exact_func_str = "100 * exp(-0.4 * x)"
            x0, y0, x_end, steps = 0.0, 100.0, 5.0, 20

    elif input_mode == "Manual Input":
        st.markdown("### :material/edit_note: Custom Equation")
        func_str = st.text_input("Equation: y' = f(x, y)", value="x + y")
        exact_func_str = st.text_input("Exact Solution: y(x) (Optional)", value="")

        with st.container():
            col1, col2 = st.columns(2)
            x0 = col1.number_input("Start (x0)", value=0.0, format="%.4f")
            y0 = col2.number_input("Initial y (y0)", value=1.0, format="%.4f")

            col3, col4 = st.columns(2)
            x_end = col3.number_input("End (x_end)", value=2.0, format="%.4f")
            steps = col4.number_input("Steps (N)", min_value=1, value=10, step=1)

    else:
        st.markdown("### :material/file_upload: Upload Data")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file", type=["csv", "xls", "xlsx", "txt"]
        )
        st.caption("Required columns: func, x0, y0, x_end, steps, exact_func")

        if uploaded_file:
            df_input = DataLoader.parse_file(uploaded_file)
            if not df_input.empty:
                # read parameters from the uploaded dataframe
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
                    f"Loaded: {uploaded_file.name}", icon=":material/check_circle:"
                )
            else:
                st.stop()
        else:
            st.warning("Waiting for file...", icon=":material/upload_file:")
            st.stop()

    st.divider()
    st.subheader(":material/layers: Select Methods")
    show_euler = st.checkbox("Euler Method", value=True)
    show_heun = st.checkbox("Heun Method", value=True)
    show_rk4 = st.checkbox("RK4 Method", value=True)
    show_exact = st.checkbox("Exact Solution", value=True)

    st.markdown("🔧 **Extras**")
    show_field = st.checkbox("Show Slope Field", value=False)

    selected_methods = [
        m
        for m, show in zip(
            ["Euler", "Heun", "RK4", "Exact"],
            [show_euler, show_heun, show_rk4, show_exact],
        )
        if show
    ]

# ==========================================================
# Equation Preview Section
# ==========================================================
st.markdown("### :material/visibility: Equation Preview")
preview_container = st.container(border=True)
with preview_container:
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        try:
            parsed_f = sp.sympify(func_str)
            st.markdown(
                "<p style='color: #6366f1; margin-bottom: 2px; font-weight: 600;'>ODE:</p>",
                unsafe_allow_html=True,
            )
            st.latex(r"\frac{dy}{dx} = " + sp.latex(parsed_f))
        except Exception:
            st.caption(":material:error: Invalid equation format.")

    with col_pre2:
        if exact_func_str and exact_func_str.lower() != "nan":
            try:
                parsed_exact = sp.sympify(exact_func_str)
                st.markdown(
                    "<p style='color: #10b981; margin-bottom: 2px; font-weight: 600;'>Exact Solution:</p>",
                    unsafe_allow_html=True,
                )
                st.latex(r"y(x) = " + sp.latex(parsed_exact))
            except Exception:
                st.caption(":material:question_mark: Exact solution failed to parse.")
        else:
            st.markdown(
                "<p style='color: #94a3b8; margin-bottom: 2px; font-weight: 600;'>Exact Solution:</p>",
                unsafe_allow_html=True,
            )
            st.caption(":material:lock: No exact solution provided.")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# Main Execution Block
# ==========================================================
if st.button(
    "Run Simulation",
    type="primary",
    use_container_width=True,
    icon=":material/rocket_launch:",
):
    try:
        # initialize solver
        solver = ODESolver(func_str, x0, y0, x_end, steps)
        x_vals = solver.get_x_grid()

        # collect results based on selected UI checkboxes
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

        # build final dataframe
        df_results = pd.DataFrame({"x": x_vals})
        for method, y_vals in results.items():
            df_results[method] = y_vals

        # compute errors if exact solution exists
        if has_exact:
            for method in ["Euler", "Heun", "RK4"]:
                if method in results:
                    df_results[f"Error_{method}"] = np.abs(
                        results["Exact"] - results[method]
                    )

        # setup results tabs
        tab1, tab2, tab3 = st.tabs(
            [
                ":material/movie: Animation",
                ":material/table_chart: Data Table",
                ":material/show_chart: Error Chart",
            ]
        )

        with tab1:
            if not selected_methods:
                st.warning(
                    "Please select at least one method to plot.",
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
                    "Download CSV",
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
                    "Download Excel",
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
                                name=f"{method} Error",
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
                    title="Error Convergence (Log Scale)",
                    xaxis=dict(
                        showgrid=True, gridcolor="rgba(255,255,255,0.04)", title="X"
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Log(|Error|)",
                    ),
                )
                st.plotly_chart(fig_err, theme=None, use_container_width=True)
            else:
                st.warning(
                    "Exact solution is needed to calculate and plot errors.",
                    icon=":material/error:",
                )

    except Exception as e:
        st.error(f"Engine Error: {str(e)}", icon=":material/bug_report:")
