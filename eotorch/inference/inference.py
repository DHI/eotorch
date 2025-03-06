from pathlib import Path
from typing import Callable

import numpy as np
import rasterio as rst
from alive_progress import alive_it
from matplotlib import pyplot as plt
from rasterio.io import BufferedDatasetWriter

from eotorch.inference import inference_utils as iu
from eotorch.plot import plot


def predict_on_tif(
    tif_file_path: str | Path,
    prediction_func: Callable,
    patch_size: int = 64,
    overlap: int = 2,
    class_mapping: dict[int, str] = None,
    func_supports_batching: bool = True,
    batch_size: int = 8,
    out_file_path: str | Path = None,
    show_results: bool = False,
    ax: plt.Axes = None,
    argmax_dim: int = -3,
) -> Path:
    """
    Predict segmentation classes on a TIF file using a custom prediction function.

    Parameters
    ----------
    tif_file_path : str | Path
        Path to the input TIF file.
    prediction_func : Callable
        Accepts a NumPy array of shape (batch_size, patch_size, patch_size, n_channels)
        and returns a NumPy array of shape (batch_size, patch_size, patch_size, n_classes).
    patch_size : int
        Integer size of the patch to use for prediction.
    overlap : int
        Overlap factor between patches (larger values increase overlap).
    classes : dict[int, str], optional
        Mapping from predicted class indices to class names for visualization.
    func_supports_batching : bool
        Whether the prediction_func supports batched processing.
    batch_size : int
        The batch size used for prediction (ignored if func_supports_batching is False).
    out_file_path : str | Path, optional
        Output path for saving the results. Writes to a "predictions" subfolder if None.
    show_results : bool
        If True, display the prediction output in a notebook environment.
    ax : plt.Axes, optional
        Matplotlib Axes object for plotting if show_results is True.

    Returns
    -------
    Path
        The path to the produced TIF file or plot visualization (when show_results=True).
    """
    tif_file_path = Path(tif_file_path)
    with rst.open(tif_file_path) as src_in:
        meta = src_in.meta.copy()
        old_no_data = meta["nodata"]

    meta.update({"dtype": "uint8", "count": 1, "nodata": 0})
    batch_size = batch_size if func_supports_batching else 1

    if out_file_path is None:
        out_file_path = (
            tif_file_path.parent / "predictions" / f"{tif_file_path.stem}_pred.tif"
        )
    out_file_path = Path(out_file_path)
    out_file_path.parent.mkdir(exist_ok=True, parents=True)

    # roughly estimate how many batches we will have
    total_windows = int(
        (meta["height"] / (patch_size / overlap))
        * (meta["width"] / (patch_size / overlap))
        / batch_size
    )

    with rst.open(out_file_path, "w", **meta) as dest:
        for batch, windows in alive_it(
            iu.patch_generator(tif_file_path, patch_size, overlap, batch_size),
            total=total_windows,
            force_tty=True,
            finalize=lambda bar: bar.title("Inference finished."),
        ):
            if (batch == old_no_data).all():
                continue
            pred = prediction_func(batch)

            # with BufferedDatasetWriter(dest) as writer:
            for i, window in enumerate(windows):
                class_pred = np.argmax(pred[i], axis=argmax_dim).astype("uint8") + 1
                unbuffered_window = iu.buffered_to_unbuffered(
                    window,
                    buffer=int(patch_size * (1 / (2 * overlap))),
                    img_height=meta["height"],
                    img_width=meta["width"],
                )
                window_arr = iu.crop_np_to_window(class_pred, window, unbuffered_window)
                dest.write_band(1, window_arr, window=unbuffered_window)

    if show_results:
        print(f"Showing results for {out_file_path}")
        with rst.open(out_file_path) as src:
            predictions = src.read(1)
            nodata_value = src.nodata
            plot.plot_numpy_array(
                array=predictions,
                class_mapping=class_mapping,
                ax=ax,
                nodata_value=nodata_value,
            )
    return out_file_path
