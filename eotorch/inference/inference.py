import itertools
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import rasterio as rst
from alive_progress import alive_it
from matplotlib import pyplot as plt

from eotorch.inference import inference_utils as iu
from eotorch.plot import plot_class_raster


def predict_on_tif_generic(
    tif_file_path: str | Path,
    prediction_func: Callable,
    data_and_window_generator: Generator = None,
    patch_size: int = 256,
    class_mapping: dict[int, str] = None,
    func_supports_batching: bool = True,
    batch_size: int = 8,
    out_file_path: str | Path = None,
    show_results: bool = False,
    ax: plt.Axes = None,
    progress_bar: bool = True,
    overlap: int = 2,
    dtype : str = "uint8",
    num_bands : int | None = None,
    nodata_value : int = 0,
) -> Path:
    """
    Predict segmentation classes on a TIF file using a custom prediction function.
    Also allows for a custom data_and_window_generator to be passed in, which can be used
    to generate patches and windows for the input data. This is useful for cases where
    the input data is not a simple TIF file, or when you want to use a custom patching strategy.
    The function will use the provided data_and_window_generator to generate batches of data and windows,
    and then apply the prediction function to each batch. The results will be written to a new TIF file.
    By default,

    Parameters
    ----------
    tif_file_path : str | Path
        Path to the input TIF file.
    prediction_func : Callable
        Accepts the input image batch supplied by the data_and_window_generator
        and returns a NumPy array of shape (batch_size, patch_size, patch_size, num_classes). 
        Dimensions and data type of the input batch will depend on the implementation of the data_and_window_generator, 
        but it should be compatible with the prediction_func.
    data_and_window_generator : Generator
        A generator that yields batches of data and their corresponding windows.
        The generator should yield tuples of the form (batch, windows), where
        windows is a list of rasterio windows (in image coordinates) indicating the location of each patch
        If None, the function will use a default patch generator, which simply loads unmodified patches straight from the TIF file. 
        Defaults to None.
    patch_size : int 
        Integer size of the patch to use for prediction. Defaults to 256.
    class_mapping : dict[int, str], optional
        Mapping from predicted class indices to class names for visualization. Defaults to None.
    func_supports_batching : bool
        Whether the prediction_func supports batched processing. Defaults to True.
    batch_size : int
        The batch size used for prediction (ignored if func_supports_batching is False). Defaults to 8.
    out_file_path : str | Path, optional
        Output path for saving the results. Writes to a "predictions" subfolder if None. Defaults to None.
    show_results : bool
        If True, display the prediction output in a notebook environment. Defaults to False.
    ax : plt.Axes, optional
        Matplotlib Axes object for plotting if show_results is True. Defaults to None, in which case a new figure and axes will be created.
    progress_bar : bool
        Whether to display a progress bar during inference. Defaults to True.
    overlap : int
        Overlap factor between patches (larger values increase overlap). Defaults to 2, which means 50% overlap.
    dtype : str
        Data type for the output TIF file. Defaults to "uint8".
    num_bands : int | None
        Number of bands in the output TIF file. If None, it will be auto-detected from the first non-nodata prediction. Defaults to None.
    nodata_value : int
        Value to use for no-data pixels in the output TIF file. Defaults to 0.


    Returns
    -------
    Path
        The path to the produced TIF file or plot visualization (when show_results=True).
    """

    tif_file_path = Path(tif_file_path)
    with rst.open(tif_file_path) as src_in:
        meta = src_in.meta.copy()
        old_no_data = meta["nodata"]

    batch_size = batch_size if func_supports_batching else 1

    if out_file_path is None:
        out_file_path = (
            tif_file_path.parent / "predictions" / f"{tif_file_path.stem}_pred.tif"
        )
    out_file_path = Path(out_file_path)
    out_file_path.parent.mkdir(exist_ok=True, parents=True)

    if data_and_window_generator is None:
        data_and_window_generator = iu.patch_generator(
            tif_file_path, patch_size, overlap, batch_size
        )

    # Auto-detect num_bands from first non-nodata prediction when not explicitly set
    _raw_gen = iter(data_and_window_generator)
    try:
        _first_batch, _first_windows = next(_raw_gen)
    except StopIteration:
        return out_file_path
    if not (_first_batch == old_no_data).all():
        _peek_pred = iu.prediction_to_numpy(prediction_func(_first_batch))
        _sample = _peek_pred[0]
        if num_bands is None:
            num_bands = _sample.shape[0] if _sample.ndim == 3 else 1
        if dtype == "uint8" and _sample.dtype != np.uint8:
            dtype = str(_sample.dtype)
    data_and_window_generator = itertools.chain([(_first_batch, _first_windows)], _raw_gen)

    meta.update({"dtype": dtype, "count": num_bands, "nodata": nodata_value})

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

    any_predictions_written = False
    try:
        with rst.open(out_file_path, "w", **meta) as dest:
            for batch, windows in (
                alive_it(
                    data_and_window_generator,
                    total=total_windows,
                    force_tty=True,
                    monitor_end=False,
                    finalize=lambda bar: bar.title("Inference finished."),
                )
                if progress_bar
                else data_and_window_generator
            ):
                if (batch == old_no_data).all():
                    continue

                pred = prediction_func(batch)

                pred = iu.prediction_to_numpy(pred)

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
                    if window_arr.ndim == 2:
                        dest.write_band(1, window_arr, window=unbuffered_window)
                    else:
                        dest.write(window_arr, window=unbuffered_window)
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
