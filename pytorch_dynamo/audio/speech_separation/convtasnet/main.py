import tarfile
import urllib.request

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend
import torch
import torchaudio
from metric import sdri
from mir_eval import separation
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX


def sisdri_metric(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mix: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        estimate = estimate - estimate.mean(axis=2, keepdim=True)
        reference = reference - reference.mean(axis=2, keepdim=True)
        mix = mix - mix.mean(axis=2, keepdim=True)

        si_sdri = sdri(estimate, reference, mix, mask=mask)

    return si_sdri.mean().item()


def postprocess(est, src, mix, mask):
    sisdri = sisdri_metric(est, src, mix, mask)
    src = src.detach().numpy()
    mix = mix.repeat(1, src.shape[1], 1).detach().numpy()
    sdr, sir, sar, _ = separation.bss_eval_sources(src[0], est[0].detach().numpy())
    sdr_mix, sir_mix, sar_mix, _ = separation.bss_eval_sources(src[0], mix[0])
    results = torch.tensor(
        [
            sdr.mean() - sdr_mix.mean(),
            sisdri,
            sir.mean() - sir_mix.mean(),
            sar.mean() - sar_mix.mean(),
        ]
    )
    return results


def main():
    # Prepare input dataset
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

    # Load and compile the model
    model = CONVTASNET_BASE_LIBRI2MIX.get_model()
    model = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": "./compiled_model"},
    )

    # (Optional) First call of forward invokes the compilation
    model(mixture)

    # Run inference using the compiled model
    estimate_sources = model(mixture)

    # Postprocessing output
    input_mask = torch.concat(
        [clean_source[None, :] != 0 for clean_source in clean_sources], dim=1
    ).float()
    clean_sources = torch.concat(
        [clean_source[None, :] for clean_source in clean_sources], dim=1
    )
    results = postprocess(estimate_sources, clean_sources, mixture, input_mask)

    # Show results
    print(f"SDR improvement: {results[0].item()}")
    print(f"Si-SDR improvement: {results[1].item()}")
    print(f"SIR improvement: {results[2].item()}")
    print(f"SAR improvement: {results[3].item()}")


if __name__ == "__main__":
    main()
