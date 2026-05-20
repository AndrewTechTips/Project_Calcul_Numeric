import numpy as np
import plotly.graph_objects as go


def generate_slope_field(f, x_range, y_range, density=20):
    """Builds the vector field background to show slope direction."""
    x_grid = np.linspace(x_range[0], x_range[1], density)
    y_grid = np.linspace(y_range[0], y_range[1], density)

    X, Y = np.meshgrid(x_grid, y_grid)
    U = np.ones_like(X)

    try:
        V = f(X, Y)
        # normalize vectors so arrows look consistent
        N = np.sqrt(U**2 + V**2)
        U, V = U / N, V / N
    except:
        # fallback to zeros if the function domain fails
        U, V = np.zeros_like(X), np.zeros_like(X)

    # scale lines for visual clarity
    length = (x_range[1] - x_range[0]) / (density * 1.5)

    x_lines, y_lines = [], []
    for i in range(density):
        for j in range(density):
            x_pos, y_pos = X[i, j], Y[i, j]
            dx, dy = U[i, j] * length, V[i, j] * length

            # create the line segment with a None gap to break the trace
            x_lines.extend([x_pos - dx / 2, x_pos + dx / 2, None])
            y_lines.extend([y_pos - dy / 2, y_pos + dy / 2, None])

    return x_lines, y_lines


def build_cinematic_plot(
    x_vals: np.ndarray, y_dict: dict, active_methods: list, f_callable, show_field: bool
) -> go.Figure:
    """Generates the interactive Plotly animation with frames."""
    fig = go.Figure()

    # UI colors for each method
    colors = {
        "Euler": "#f43f5e",
        "Heun": "#3b82f6",
        "RK4": "#10b981",
        "Exact": "#f8fafc",
    }

    methods_to_plot = [m for m in active_methods if m in y_dict]

    # optional: render the background vector field
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

    # setup initial data points
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

    # create animation frames
    frames = []
    for k in range(1, len(x_vals) + 1):
        frame_data = []
        if show_field:
            frame_data.append(go.Scatter(x=x_lines, y=y_lines))
        for method in methods_to_plot:
            frame_data.append(go.Scatter(x=x_vals[:k], y=y_dict[method][:k]))

        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig.frames = frames

    # setup play/pause controls and sliders
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
            text="Numerical Integration Progress", font=dict(size=22, color="#ffffff")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            title="X Axis",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            title="Y Axis",
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
