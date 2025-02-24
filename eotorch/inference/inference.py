from pathlib import Path
from typing import Callable

import numpy as np
import rasterio as rst
from alive_progress import alive_it
from matplotlib import pyplot as plt
from rasterio.io import BufferedDatasetWriter

from eotorch.inference import inference_utils as iu


def predict_on_tif(
    tif_file_path: str | Path,
    prediction_func: Callable,
    patch_size: int = 64,
    overlap: int = 2,
    classes: dict[int, str] = None,
    func_supports_batching: bool = True,
    batch_size: int = 8,
    out_file_path: str | Path = None,
    show_results: bool = False,
    ax: plt.Axes = None,
) -> Path:
    """
    Predict on a tif file using a prediction function.

    Parameters
    ----------
        tif_file_path: path to tif file
        prediction_func: prediction function to call to get model predictions. The function should accept
            a numpy array of shape (batch_size, patch_size, patch_size, n_channels) and return a numpy array
            of shape (batch_size, patch_size, patch_size, n_classes)
        patch_size: size of the patches to use for prediction
        classes: mapping of prediction indices to class names
        func_supports_batching: whether the prediction function supports batching
        batch_size: batch size to use for prediction, ignored if func_supports_batching is False
        out_file_path: path to save the results to, ignored if save_results is False. If None, the results
            will be saved to config.DATA_DIR / "predictions" under the same name as the input tif file
        show_results: whether to show the results in a notebook environment
        ax: matplotlib axes to plot the results on, ignored if show_results is False
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
                class_pred = np.argmax(pred[i], axis=-1).astype("uint8") + 1
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
        return iu.plot_predictions_pyplot(out_file_path, classes, ax=ax)
    return out_file_path
