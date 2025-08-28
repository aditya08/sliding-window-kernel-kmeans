import os
import bz2
import argparse
import requests

datasets = {
    "acoustic": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/acoustic_scale.bz2",
    "cifar10": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.bz2",
    "ledgar": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/ledgar_lexglue_tfidf_train.svm.bz2",
    "letter": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale",
    "mnist": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2",
    "scotus": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/scotus_lexglue_tfidf_train.svm.bz2"
}

def download_file(url, destination, remove_bz2=False):
    """
    Downloads a file from a given URL to a specified destination.

    Args:
        url (str): The URL of the file to download.
        destination (str): The local path where the file should be saved.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    with open(destination, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {url} to {destination}")

    if url.endswith('.bz2'):
        with bz2.open(destination, 'rb') as compressed_file:
            decompressed_data = compressed_file.read()
            with open(destination[:-4], 'wb') as decompressed_file:
                if isinstance(decompressed_data, str):
                    decompressed_file.write(decompressed_data.encode())
                else:
                    decompressed_file.write(decompressed_data)
        if remove_bz2:
            os.remove(destination)
            print(f"Removed compressed file {destination}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for Kernel K-means.")
    parser.add_argument("dataset", choices=list(datasets.keys()) + ["all"], nargs="?", default="all", help="Name of the dataset to download (or 'all' for all datasets)")
    parser.add_argument("--destination", type=str, default="../data", help="Destination directory for the downloaded file")
    parser.add_argument("--remove-bz2", action="store_true", help="Remove .bz2 files after decompression")

    args = parser.parse_args()

    if not args.destination.endswith('/'):
        args.destination += '/'

    destination_path = args.destination + args.dataset + ('.bz2' if args.dataset in datasets else '')

    if args.dataset == 'all':
        for dataset_name, url in datasets.items():
            print(f"Downloading {dataset_name} dataset...")
            download_file(url, args.destination + dataset_name + ('.bz2' if url.endswith('.bz2') else ''), remove_bz2=args.remove_bz2)
    else:
        if args.dataset not in datasets:
            raise ValueError(f"Dataset '{args.dataset}' is not available. Choose from {list(datasets.keys())}.")
        print(f"Downloading {args.dataset} dataset...")
        download_file(datasets[args.dataset], destination_path, remove_bz2=args.remove_bz2)

if __name__ == "__main__":
    main()