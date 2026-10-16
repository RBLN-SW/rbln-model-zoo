import argparse
import os

from optimum.rbln import RBLNKandinskyV22Pipeline, RBLNKandinskyV22PriorPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="red cat, 4k photo",
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
    decoder_pipe = RBLNKandinskyV22Pipeline.from_pretrained(
        model_id=os.path.basename(decoder_model_id),
        export=False,
    )

    image_emb, zero_image_emb = prior_pipe(prompt, return_dict=False)

    # Generate image
    image = decoder_pipe(
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=50,
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
