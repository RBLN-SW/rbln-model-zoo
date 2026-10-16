import tarfile
import urllib

import rebel
import torch
import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX


def main():
    model = CONVTASNET_BASE_LIBRI2MIX.get_model()

    dataset_url = (
        "https://rbln-public.s3.ap-northeast-2.amazonaws.com/datasets/Libri2Mix.tar"
    )
    dataset_path = "./Libri2Mix.tar"
    with urllib.request.urlopen(dataset_url) as response, open(dataset_path, "wb") as f:
        f.write(response.read())
    tar = tarfile.open(dataset_path)
    tar.extractall(".")

    dataset = torchaudio.datasets.LibriMix(".", subset="test")
    sample_rate, mixture, clean_sources = dataset[0]
    mixture = mixture.reshape(1, 1, -1)

    input_info = [
        ("input_np", list(mixture.shape), torch.float32),  # maximum frame length
    ]

    compiled_model = rebel.compile_from_torch(model, input_info)
    compiled_model.save("convtasnet_libri2mix.rbln")


if __name__ == "__main__":
    main()
