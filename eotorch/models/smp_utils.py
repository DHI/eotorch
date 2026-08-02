def decoder_channels_from_num_filters(num_filters: int, encoder_depth: int) -> list[int]:
    """Derive a decreasing decoder_channels list for segmentation_models_pytorch decoders.

    num_filters is treated as the maximum (first-stage) number of filters, and each
    subsequent stage halves it. The length of the returned list is equal to encoder_depth.
    """
    return [num_filters // (2**i) for i in range(encoder_depth)]
