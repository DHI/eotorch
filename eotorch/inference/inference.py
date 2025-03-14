from pathlib import Path
from typing import Callable

import numpy as np
import rasterio as rst
from alive_progress import alive_it
from matplotlib import pyplot as plt

from eotorch.inference import inference_utils as iu
from eotorch.plot import plot_class_raster


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
) -> Path:
    """
    Predict segmentation classes on a TIF file using a custom prediction function.

    Parameters
    ----------
    tif_file_path : str | Path
        Path to the input TIF file.
    prediction_func : Callable
        Accepts a NumPy array of shape (batch_size, patch_size, patch_size, n_channels)
        and returns a NumPy array of shape (batch_size, patch_size, patch_size, num_classes).
    patch_size : int
        Integer size of the patch to use for prediction.
    overlap : int
        Overlap factor between patches (larger values increase overlap).
    class_mapping : dict[int, str], optional
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

    def _plot(data_window_only: bool):
        print(f"Showing results for {out_file_path}")
        return plot_class_raster(
            tif_file_path=out_file_path,
            class_mapping=class_mapping,
            ax=ax,
            data_window_only=data_window_only,
        )

    any_predictions_written, nan_inputs_logged = False, False
    try:
        with rst.open(out_file_path, "w", **meta) as dest:
            for batch, windows in alive_it(
                iu.patch_generator(tif_file_path, patch_size, overlap, batch_size),
                total=total_windows,
                force_tty=True,
                monitor_end=False,
                finalize=lambda bar: bar.title("Inference finished."),
            ):
                if (batch == old_no_data).all():
                    continue

                if not nan_inputs_logged:
                    if np.isnan(batch).any():
                        print(
                            f"NaN values found in the input image which are not equal to the nodata value. They will be replaced with the nodata value {old_no_data}."
                        )
                        nan_inputs_logged = True
                # handle case of there being nan values that are not set to nodata
                batch = np.nan_to_num(batch, nan=old_no_data)

                pred = prediction_func(batch)

                for i, window in enumerate(windows):
                    class_pred = pred[i]
                    unbuffered_window = iu.buffered_to_unbuffered(
                        window,
                        buffer=int(patch_size * (1 / (2 * overlap))),
                        img_height=meta["height"],
                        img_width=meta["width"],
                    )
                    window_arr = iu.crop_np_to_window(
                        class_pred, window, unbuffered_window
                    )
                    dest.write_band(1, window_arr, window=unbuffered_window)
                    if not any_predictions_written:
                        any_predictions_written = True

    except KeyboardInterrupt:
        if show_results and any_predictions_written:
            print(
                "Inference interrupted. Showing predictions that have been generated so far."
            )
            return _plot(data_window_only=True)
        print("Inference interrupted.")
        return

    if show_results:
        return _plot(data_window_only=False)
    return out_file_path
