import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import io

# ==========================================
# 1. Page Configuration & Ultra-Premium Styling
# ==========================================
st.set_page_config(page_title="ODE Numerical Solver", layout="wide", page_icon="🌌")

# CSS Injection for Animated Gradient & Deep Glassmorphism
# Safe usage: Only static CSS strings are injected, no user-generated DOM components.
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

    /* Header Transparency */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Premium Input Elements with Glow Focus */
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

    /* Refined Containers & Expanders */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.015);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
    }

    /* Cyber/Neon Primary Buttons */
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


# ==========================================
# 2. Mathematical Core
# ==========================================
class ODESolver:
    def __init__(self, func_str: str, x0: float, y0: float, x_end: float, steps: int):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x_end = float(x_end)
        self.steps = int(steps)
        self.h = (self.x_end - self.x0) / self.steps

        self.x_sym, self.y_sym = sp.symbols("x y")
        try:
            self.expr = sp.sympify(func_str)
            self.f = sp.lambdify((self.x_sym, self.y_sym), self.expr, modules=["numpy"])
        except Exception as e:
            raise ValueError(f"Syntax error in function parsing: {e}")

    def get_x_grid(self) -> np.ndarray:
        return np.linspace(self.x0, self.x_end, self.steps + 1)

    def solve_euler(self) -> np.ndarray:
        y_vals = np.zeros(self.steps + 1, dtype=np.float64)
        y_vals[0] = self.y0
        x_vals = self.get_x_grid()
        for i in range(self.steps):
            y_vals[i + 1] = y_vals[i] + self.h * self.f(x_vals[i], y_vals[i])
        return y_vals

    def solve_heun(self) -> np.ndarray:
        y_vals = np.zeros(self.steps + 1, dtype=np.float64)
        y_vals[0] = self.y0
        x_vals = self.get_x_grid()
        for i in range(self.steps):
            k1 = self.h * self.f(x_vals[i], y_vals[i])
            k2 = self.h * self.f(x_vals[i] + self.h, y_vals[i] + k1)
            y_vals[i + 1] = y_vals[i] + 0.5 * (k1 + k2)
        return y_vals

    def solve_rk4(self) -> np.ndarray:
        y_vals = np.zeros(self.steps + 1, dtype=np.float64)
        y_vals[0] = self.y0
        x_vals = self.get_x_grid()
        for i in range(self.steps):
            xi, yi = x_vals[i], y_vals[i]
            k1 = self.h * self.f(xi, yi)
            k2 = self.h * self.f(xi + self.h / 2, yi + k1 / 2)
            k3 = self.h * self.f(xi + self.h / 2, yi + k2 / 2)
            k4 = self.h * self.f(xi + self.h, yi + k3)
            y_vals[i + 1] = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y_vals


class DataLoader:
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


# ==========================================
# 3. Vector Field & Cinematic Plot Engine
# ==========================================
def generate_slope_field(f, x_range, y_range, density=20):
    """Calculates the vector field for the background to show ODE behavior."""
    x_grid = np.linspace(x_range[0], x_range[1], density)
    y_grid = np.linspace(y_range[0], y_range[1], density)

    X, Y = np.meshgrid(x_grid, y_grid)
    U = np.ones_like(X)

    try:
        V = f(X, Y)
        # Normalize vectors for consistent arrow lengths
        N = np.sqrt(U**2 + V**2)
        U, V = U / N, V / N
    except:
        U, V = np.zeros_like(X), np.zeros_like(X)

    # Scale length for visual appeal
    length = (x_range[1] - x_range[0]) / (density * 1.5)

    x_lines, y_lines = [], []
    for i in range(density):
        for j in range(density):
            x_pos, y_pos = X[i, j], Y[i, j]
            dx, dy = U[i, j] * length, V[i, j] * length
            x_lines.extend([x_pos - dx / 2, x_pos + dx / 2, None])
            y_lines.extend([y_pos - dy / 2, y_pos + dy / 2, None])

    return x_lines, y_lines


def build_cinematic_plot(
    x_vals: np.ndarray, y_dict: dict, active_methods: list, f_callable, show_field: bool
) -> go.Figure:
    fig = go.Figure()

    colors = {
        "Euler": "#f43f5e",  # Rose
        "Heun": "#3b82f6",  # Blue
        "RK4": "#10b981",  # Emerald
        "Exact": "#f8fafc",  # White
    }

    methods_to_plot = [m for m in active_methods if m in y_dict]

    # Optional: Draw Slope Field in background
    if show_field:
        y_all_vals = [val for method in methods_to_plot for val in y_dict[method]]
        y_min, y_max = min(y_all_vals) - 0.5, max(y_all_vals) + 0.5
        x_lines, y_lines = generate_slope_field(
            f_callable, (x_vals[0], x_vals[-1]), (y_min, y_max)
        )

        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.15)", width=1.5),
                name="Slope Field",
                hoverinfo="skip",
            )
        )

    # Initialize dynamic lines
    for method in methods_to_plot:
        fig.add_trace(
            go.Scatter(
                x=[x_vals[0]],
                y=[y_dict[method][0]],
                mode="lines+markers",
                name=method,
                line=dict(color=colors.get(method, "#ffffff"), width=3),
                marker=dict(
                    size=8, symbol="circle", line=dict(width=1, color="rgba(0,0,0,0.5)")
                ),
            )
        )

    # Construct Frames for Playback
    frames = []
    for k in range(1, len(x_vals) + 1):
        frame_data = []
        if show_field:
            frame_data.append(
                go.Scatter(x=x_lines, y=y_lines)
            )  # Keep background static
        for method in methods_to_plot:
            frame_data.append(go.Scatter(x=x_vals[:k], y=y_dict[method][:k]))
        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                bgcolor="rgba(15, 23, 42, 0.8)",
                font=dict(color="#ffffff", size=14, family="sans-serif"),
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True), fromcurrent=True
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
                        ],
                    ),
                ],
                direction="left",
                pad={"r": 10, "t": 87},
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top",
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(k)],
                            dict(
                                mode="immediate", frame=dict(duration=100, redraw=True)
                            ),
                        ],
                        label=str(k),
                    )
                    for k in range(1, len(x_vals) + 1)
                ],
                x=0.1,
                y=0,
                len=0.9,
                xanchor="left",
                yanchor="top",
                font=dict(color="#ffffff"),
                bgcolor="rgba(15, 23, 42, 0.8)",
                activebgcolor="#4f46e5",
            )
        ],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        title=dict(
            text="Numerical Integration with Phase Analysis",
            font=dict(size=22, color="#ffffff"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            title="Domain (X)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            title="Solution (Y)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(10, 15, 25, 0.7)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color="#ffffff"),
        ),
    )
    return fig


# ==========================================
# 4. Interface & Routing
# ==========================================
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
    show_field = st.checkbox(
        "Display Slope Vector Field",
        value=True,
        help="Plots the derivative vector field f(x,y) in the background.",
    )

    selected_methods = []
    if show_euler:
        selected_methods.append("Euler")
    if show_heun:
        selected_methods.append("Heun")
    if show_rk4:
        selected_methods.append("RK4")
    if show_exact:
        selected_methods.append("Exact")

# ==========================================
# 5. Execution Logic & Dashboard
# ==========================================
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
            exact_expr = sp.sympify(exact_func_str)
            exact_f = sp.lambdify(x_sym, exact_expr, "numpy")
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

        # Setup Tabs for a cleaner, dashboard-like look
        tab1, tab2, tab3 = st.tabs(
            ["🎥 Visual Animation", "📊 Data Matrix", "📉 Error Analysis (Log Scale)"]
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
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export CSV",
                    data=csv_data,
                    file_name="ode_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_dl2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_results.to_excel(writer, index=False, sheet_name="Data")
                excel_data = output.getvalue()
                st.download_button(
                    "📥 Export Excel",
                    data=excel_data,
                    file_name="ode_results.xlsx",
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
                                line=dict(
                                    color=colors.get(f"Error_{method}", "#fff"), width=2
                                ),
                            )
                        )

                fig_err.update_layout(
                    title="Absolute Error Convergence (Log Scale)",
                    yaxis_type="log",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.05)",
                        title="Log(|Error|)",
                    ),
                )
                st.plotly_chart(fig_err, theme=None, use_container_width=True)
                st.info(
                    "💡 **Insight:** Observați diferența dintre metoda RK4 (Ordinul 4) și Euler (Ordinul 1). Scara logaritmică evidențiază magnitudinea preciziei superioare aduse de RK4."
                )
            else:
                st.warning(
                    "Pentru analiza erorilor, este necesară soluția exactă (Exact Solution)."
                )

    except Exception as e:
        st.error(f"Computation Error: {str(e)}")
