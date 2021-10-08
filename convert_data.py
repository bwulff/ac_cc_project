import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

def main():
	
	# read the CSV file
	data = pd.read_csv("data/eeg-data.csv")
	N = data.shape[0]

	# print("collecting labels")
	# labels = {i:label for i, label in enumerate(set(data["label"]))}
	# print(labels)
	
	print("Filtering samples with signal_quality < 128")
	data_filtered = []
	for _, row in tqdm(data.iterrows()):
		if row["signal_quality"] < 128:
			x = np.array(eval(row["raw_values"]))
			# x = (x + 2048) / 4096
			vmin = np.min(x)
			vmax = np.max(x)
			x = (x - vmin) / (vmax - vmin)
			data_filtered.append(x)

	print(f"Found {len(data_filtered)} samples.")

	# # prepare the H5 file
	print("Saving data")
	with h5py.File("data/eeg-filtered-normalized.h5", "w") as f:
		f.create_dataset("data", (len(data_filtered), 512), dtype='float')
		for i, sample in tqdm(enumerate(data_filtered)):
			f["data"][i] = sample 
	
if __name__ == "__main__":
	main()