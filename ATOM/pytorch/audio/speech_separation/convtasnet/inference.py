# BSD 2-Clause License

# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tarfile
import urllib.request

import rebel
import torch
import torchaudio
from metric import sdri
from mir_eval import separation


# ref: https://github.com/pytorch/audio/blob/7f6209b44a1b838e9f33fdd382a3c4ae14e8297f/examples/source_separation/lightning_train.py#L25
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


# ref: https://github.com/pytorch/audio/blob/7f6209b44a1b838e9f33fdd382a3c4ae14e8297f/examples/source_separation/eval.py#L9
def postprocess(est, src, mix, mask):
    sisdri = sisdri_metric(torch.tensor(est), src, mix, mask)
    src = src.detach().numpy()
    mix = mix.repeat(1, src.shape[1], 1).detach().numpy()
    sdr, sir, sar, _ = separation.bss_eval_sources(src[0], est[0])
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

    # load model
    module = rebel.Runtime("convtasnet_libri2mix.rbln")

    # run
    estimate_sources = module.run(mixture.numpy())

    # postprocessing output
    input_mask = torch.concat(
        [clean_source[None, :] != 0 for clean_source in clean_sources], dim=1
    ).float()
    clean_sources = torch.concat(
        [clean_source[None, :] for clean_source in clean_sources], dim=1
    )
    results = postprocess(estimate_sources, clean_sources, mixture, input_mask)

    # show results
    print(f"SDR improvement: {results[0].item()}")
    print(f"Si-SDR improvement: {results[1].item()}")
    print(f"SIR improvement: {results[2].item()}")
    print(f"SAR improvement: {results[3].item()}")


if __name__ == "__main__":
    main()
