"""Unified CLI entry point. Run ``uv run datatonfach --help``."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="datatonfach: fire-risk pipeline over satellite imagery.")


@app.command()
def compose(folders: list[str] = typer.Option(None, help="Folders under data/ to process.")) -> None:
    """Build composed false-RGB GeoTIFFs from elevation + NDVI bands."""
    from .features.input_images import build_composed_images

    build_composed_images(folders)


@app.command()
def featurize(
    folders: list[str] = typer.Option(None, help="Training folders under data/."),
    backbone: str = typer.Option("base", help="DinoV2 backbone size: small|base|large|giant."),
) -> None:
    """Compute DinoV2 features + labels for training."""
    from .features.dataset import build_training_dataset

    build_training_dataset(train_folders=folders, backbone_size=backbone)


@app.command()
def train(k: int = 11, n_splits: int = 5, show_plots: bool = False) -> None:
    """Train a FAISS KNN classifier with stratified k-fold CV."""
    from .models.train import train as _train

    _train(k=k, n_splits=n_splits, show_plots=show_plots)


@app.command()
def predict(
    raster: Path = typer.Option(None, help="Input composed.tif path. Defaults to data/valpo/composed.tif."),
    backbone: str = typer.Option("base"),
) -> None:
    """Predict a fire-risk class map over a composed raster."""
    from .models.predict import predict_default, predict_raster

    out = predict_raster(raster, backbone_size=backbone) if raster else predict_default()
    typer.echo(f"Saved prediction to: {out}")


@app.command()
def validate(bbox: Path = typer.Option(None), prediction: Path = typer.Option(None)) -> None:
    """Compare a predicted raster against bbox ground truth."""
    from .eval.validation import run_validation

    run_validation(bbox_path=bbox, pred_raster_path=prediction)


if __name__ == "__main__":
    app()
