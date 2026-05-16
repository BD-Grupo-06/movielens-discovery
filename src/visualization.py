from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def save_plot(fig, path: Path) -> None:
    """Save a Plotly figure as HTML and PNG."""
    fig.write_html(str(path))
    png_path = path.with_suffix(".png")
    try:
        fig.write_image(str(png_path))
    except Exception:
        # Static image export might fail if kaleido/orca is not properly configured
        pass

def plot_cumulative_variance(variance_df, title, x="component", y="cumulative_variance"):
    fig = px.line(
        variance_df,
        x=x,
        y=y,
        markers=True,
        title=title,
    )
    return fig

def plot_tsne_scatter(tsne_df, title, x="tsne_1", y="tsne_2"):
    fig = px.scatter(
        tsne_df,
        x=x,
        y=y,
        title=title,
        opacity=0.6,
    )
    return fig
