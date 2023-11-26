import torch
from models import skip

if __name__ == "__main__":
    pad = "zero"  # 'refection'
    model = skip(
        32,
        3,
        num_channels_down=[16, 32, 64, 128, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4, 4],
        filter_size_down=[7, 7, 5, 5, 3, 3],
        filter_size_up=[7, 7, 5, 5, 3, 3],
        upsample_mode="nearest",
        downsample_mode="avg",
        need_sigmoid=True,
        pad=pad,
        act_fun="LeakyReLU",
    )

    input = torch.randn(1, 32, 256, 256)
    try:
        torch.export.export(model, (input,))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e
