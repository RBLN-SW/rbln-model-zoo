import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNKandinskyV22Img2ImgPipeline, RBLNKandinskyV22PriorPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="A red cartoon frog, 4k",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    prior_model_id = "kandinsky-community/kandinsky-2-2-prior"
    decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"
    prompt = args.prompt

    # Load compiled model
    prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
        model_id=os.path.basename(prior_model_id),
        export=False,
    )
    decoder_pipe = RBLNKandinskyV22Img2ImgPipeline.from_pretrained(
        model_id=os.path.basename(decoder_model_id),
        export=False,
    )

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/frog.png"
    init_image = load_image(img_url)

    image_emb, zero_image_emb = prior_pipe(prompt, return_dict=False)

    # Generate image
    image = decoder_pipe(
        image=init_image,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=100,
        strength=0.2,
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
