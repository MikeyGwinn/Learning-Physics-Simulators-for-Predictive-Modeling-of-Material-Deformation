import numpy as np

path = r"params_coarse.npz"

with np.load(path, allow_pickle=False) as data:
    print("Keys:", data.files)

    for name in data.files:
        array = data[name]

        print(f"\n{name}")
        print("  shape:", array.shape)
        print("  dtype:", array.dtype)
        print("  range:", array.min(), "to", array.max())
        print("  first values:", array.reshape(-1)[:8])