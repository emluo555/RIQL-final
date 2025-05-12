## Visualize the data corruption

import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_data(file_path, original_path, max_keys=10, max_items=5, plot_key=None):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nOpening dataset: {file_path}")
            keys = list(f.keys())
            
            for key in keys[:max_keys]: # inspect data
                print(f"\nKey: '{key}'")
                try:
                    data = f[key]

                    if isinstance(data, h5py.Group):
                        for subkey in list(data.keys())[:max_keys]:
                            print(f"{subkey}")
                    elif isinstance(data, h5py.Dataset):
                        print(f"Shape: {data.shape}")
                        print(f"Dtype: {data.dtype}")
                        if data.shape:
                            preview = data[:min(max_items, data.shape[0])]
                            print(f"First {len(preview)} item(s): {preview}")
                        else:
                            print(f"Scalar values: {data[()]}")
                    else:
                        print("Unknown type.")
                except Exception as e:
                    print(f"Couldn't read this key: {e}")

            if plot_key:
                if plot_key in f:
                    corrupt_data = f[plot_key][:]
                    plt.plot(corrupt_data[:1000], label=f"Corrupted")
                else:
                    print(f"Key '{plot_key}' not found in {file_path}")
                    return

                try:
                    with h5py.File(original_path, 'r') as original_file:
                        if plot_key in original_file:
                            original_data = original_file[plot_key][:]

                            plt.plot(original_data[:1000], label="Clean", linestyle='--')

                            print(f"Original {plot_key} mean: ", np.mean(original_data, axis=0))
                            print(f"Original {plot_key} var:  ", np.var(original_data, axis=0))

                            print(f"Corrupted {plot_key} mean:", np.mean(main_data, axis=0))
                            print(f"Corrupted {plot_key} var: ", np.var(main_data, axis=0))

                        else:
                            print(f"Key '{plot_key}' not found in original file {original_path}")
                except Exception as e:
                    print(f"Could not open original file or read data: {e}")

                plt.title(f"Corrupted {plot_key} vs Original")
                plt.xlabel("Datapoint")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{plot_key}_plot.png')
                print(f"Plot saved as '{plot_key}_plot.png'")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error opening file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupt_path", help="corrupted file path")
    parser.add_argument("--original" , help="original file path")
    parser.add_argument("--plot_key", type=str, help="Key of the dataset to plot (e.g., 'rewards')")

    args = parser.parse_args()
    visualize_data(args.corrupt_path, args.original, plot_key= args.plot_key)
