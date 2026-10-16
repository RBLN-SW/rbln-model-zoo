import os

import torch
from optimum.rbln import RBLNAutoModelForSeq2SeqLM
from transformers import PreTrainedTokenizerFast


def main():
    model_id = "gogamza/kobart-summarization"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)

    # Load compiled model
    model = RBLNAutoModelForSeq2SeqLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    text = "과거를 떠올려보자. 방송을 보던 우리의 모습을. 독보적인 매체는 TV였다. 온 가족이 둘러앉아 TV를 봤다. 간혹 가족들끼리 뉴스와 드라마, 예능 프로그램을 둘러싸고 리모컨 쟁탈전이 벌어지기도  했다. 각자 선호하는 프로그램을 ‘본방’으로 보기 위한 싸움이었다. TV가 한 대인지 두 대인지 여부도 그래서 중요했다. 지금은 어떤가. ‘안방극장’이라는 말은 옛말이 됐다. TV가 없는 집도 많다. 미디어의 혜 택을 누릴 수 있는 방법은 늘어났다. 각자의 방에서 각자의 휴대폰으로, 노트북으로, 태블릿으로 콘텐츠 를 즐긴다."  # noqa: E501

    # Prepare inputs
    inputs = tokenizer(text, padding="max_length", max_length=1024)

    generation_kwargs = {}
    generation_kwargs["num_beams"] = 1

    # Generate tokens
    output_sequence = model.generate(
        input_ids=torch.tensor([inputs.input_ids]),
        attention_mask=torch.tensor([inputs.attention_mask]),
        max_length=1024,
        **generation_kwargs,
    )

    # Show text and result
    print(f"Text: {text}")
    print(
        f"Result: {tokenizer.decode(output_sequence.squeeze().tolist(), skip_special_tokens=True)}"
    )


if __name__ == "__main__":
    main()
