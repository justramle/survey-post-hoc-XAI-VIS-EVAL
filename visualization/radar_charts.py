import plotly.graph_objects as go


def plot_radar(df_transposed, metrics, title, colors, save_path=None):
    """df_transposed: DataFrame indexed by metric name, columns = method names."""
    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatterpolar(
                r=df_transposed.loc[metric].values,
                theta=df_transposed.columns.tolist(),
                fill="toself",
                line=dict(color=colors[i % len(colors)]),
                name=metric,
            )
        )

    fig.update_layout(
        polar=dict(
            domain=dict(x=[0, 1], y=[0, 1]),
            radialaxis=dict(visible=True, tickfont=dict(size=9)),
            angularaxis=dict(ticktext=df_transposed.columns.tolist()),
        ),
        showlegend=True,
        title={"text": title, "x": 0.45, "xanchor": "center", "y": 0.87, "yanchor": "top"},
        margin=dict(l=100, b=0, t=90),
        width=600,
        height=500,
    )

    if save_path:
        fig.write_image(save_path, scale=3)
    return fig
