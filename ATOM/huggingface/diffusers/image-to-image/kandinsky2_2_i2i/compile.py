import os

import torch
from optimum.rbln import RBLNKandinskyV22Img2ImgPipeline, RBLNKandinskyV22PriorPipeline


def main():
    prior_model_id = "kandinsky-community/kandinsky-2-2-prior"
    decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"

    # Compile and export
    prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
        prior_model_id,
        torch_dtype=torch.float16,
        export=True,  # export a PyTorch model to RBLN model with optimum
    )

    decoder_pipe = RBLNKandinskyV22Img2ImgPipeline.from_pretrained(
        decoder_model_id,
        torch_dtype=torch.float16,
        export=True,
        rbln_config={"img_height": 768, "img_width": 768, "unet": {"batch_size": 2}},
    )

    # Save compiled results to disk
    prior_pipe.save_pretrained(os.path.basename(prior_model_id))
    decoder_pipe.save_pretrained(os.path.basename(decoder_model_id))


if __name__ == "__main__":
    main()
