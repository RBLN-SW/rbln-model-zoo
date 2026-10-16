import argparse
import urllib.request

import numpy as np
import rebel  # RBLN Runtime
import tf_keras.applications as app
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image

input_size_map = {
    "EfficientNetB1": 240,
    "EfficientNetB2": 260,
    "EfficientNetB3": 300,
    "EfficientNetB4": 380,
    "EfficientNetB5": 456,
    "EfficientNetB6": 528,
    "EfficientNetB7": 600,
    "EfficientNetV2B1": 240,
    "EfficientNetV2B2": 260,
    "EfficientNetV2B3": 300,
    "EfficientNetV2S": 384,
    "EfficientNetV2M": 480,
    "EfficientNetV2L": 480,
    "NASNetLarge": 331,
    "InceptionV3": 299,
    "InceptionResNetV2": 299,
    "Xception": 299,
}
preprocess_map = {
    "ResNet50": "resnet",
    "ResNet101": "resnet",
    "ResNet152": "resnet",
    "ResNet50V2": "resnet_v2",
    "ResNet101V2": "resnet_v2",
    "ResNet152V2": "resnet_v2",
    "EfficientNetB0": "efficientnet",
    "EfficientNetB1": "efficientnet",
    "EfficientNetB2": "efficientnet",
    "EfficientNetB3": "efficientnet",
    "EfficientNetB4": "efficientnet",
    "EfficientNetB5": "efficientnet",
    "EfficientNetB6": "efficientnet",
    "EfficientNetB7": "efficientnet",
    "EfficientNetV2B0": "efficientnet_v2",
    "EfficientNetV2B1": "efficientnet_v2",
    "EfficientNetV2B2": "efficientnet_v2",
    "EfficientNetV2B3": "efficientnet_v2",
    "EfficientNetV2L": "efficientnet_v2",
    "EfficientNetV2M": "efficientnet_v2",
    "EfficientNetV2S": "efficientnet_v2",
    "MobileNet": "mobilenet",
    "MobileNetV2": "mobilenet_v2",
    "MobileNetV3Small": "mobilenet_v3",
    "MobileNetV3Large": "mobilenet_v3",
    "DenseNet121": "densenet",
    "DenseNet169": "densenet",
    "DenseNet201": "densenet",
    "VGG16": "vgg16",
    "VGG19": "vgg19",
    "ConvNeXtTiny": "convnext",
    "ConvNeXtSmall": "convnext",
    "ConvNeXtBase": "convnext",
    "ConvNeXtLarge": "convnext",
    "ConvNeXtXLarge": "convnext",
    "RegNetX002": "regnet",
    "RegNetX004": "regnet",
    "RegNetX006": "regnet",
    "RegNetX008": "regnet",
    "RegNetX016": "regnet",
    "RegNetX032": "regnet",
    "RegNetX040": "regnet",
    "RegNetX064": "regnet",
    "RegNetX080": "regnet",
    "RegNetX120": "regnet",
    "RegNetX160": "regnet",
    "RegNetX320": "regnet",
    "RegNetY002": "regnet",
    "RegNetY004": "regnet",
    "RegNetY006": "regnet",
    "RegNetY008": "regnet",
    "RegNetY016": "regnet",
    "RegNetY032": "regnet",
    "RegNetY040": "regnet",
    "RegNetY064": "regnet",
    "RegNetY080": "regnet",
    "RegNetY120": "regnet",
    "RegNetY160": "regnet",
    "RegNetY320": "regnet",
    "ResNetRS50": "resnet_rs",
    "ResNetRS101": "resnet_rs",
    "ResNetRS152": "resnet_rs",
    "ResNetRS200": "resnet_rs",
    "ResNetRS270": "resnet_rs",
    "ResNetRS350": "resnet_rs",
    "ResNetRS420": "resnet_rs",
    "NASNetMobile": "nasnet",
    "NASNetLarge": "nasnet",
    "InceptionV3": "inception_v3",
    "InceptionResNetV2": "inception_resnet_v2",
    "Xception": "xception",
}


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="EfficientNetB0",
        choices=[
            "EfficientNetB0",
            "EfficientNetB1",
            "EfficientNetB2",
            "EfficientNetB3",
            "EfficientNetB4",
            "EfficientNetB5",
            "EfficientNetB6",
            "EfficientNetB7",
            "ResNet50",
            "ResNet101",
            "ResNet152",
            "ResNet50V2",
            "ResNet101V2",
            "ResNet152V2",
            "EfficientNetV2B0",
            "EfficientNetV2B1",
            "EfficientNetV2B2",
            "EfficientNetV2B3",
            "EfficientNetV2S",
            "EfficientNetV2M",
            "EfficientNetV2L",
            "DenseNet121",
            "DenseNet169",
            "DenseNet201",
            "NASNetMobile",
            "NASNetLarge",
            "InceptionV3",
            "ConvNeXtBase",
            "ConvNeXtLarge",
            "ConvNeXtSmall",
            "ConvNeXtTiny",
            "ConvNeXtXLarge",
            "RegNetX002",
            "RegNetX004",
            "RegNetX006",
            "RegNetX008",
            "RegNetX016",
            "RegNetX032",
            "RegNetX040",
            "RegNetX064",
            "RegNetX080",
            "RegNetX120",
            "RegNetX160",
            "RegNetX320",
            "RegNetY002",
            "RegNetY004",
            "RegNetY006",
            "RegNetY008",
            "RegNetY016",
            "RegNetY032",
            "RegNetY040",
            "RegNetY064",
            "RegNetY080",
            "RegNetY120",
            "RegNetY160",
            "RegNetY320",
            "VGG16",
            "VGG19",
            "InceptionResNetV2",
            "ResNetRS50",
            "ResNetRS101",
            "ResNetRS152",
            "MobileNet",
            "MobileNetV2",
            "MobileNetV3Large",
            "MobileNetV3Small",
            "ResNetRS200",
            "ResNetRS270",
            "ResNetRS350",
            "ResNetRS420",
            "Xception",
        ],
        help="(str) type, tensorflow keras model name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Prepare input image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())

    preprocess = getattr(app, preprocess_map[model_name]).preprocess_input
    input_size = input_size_map.get(model_name, 224)
    img = image.load_img(img_path, target_size=(224, 224)).resize(
        (input_size, input_size)
    )

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess(x)
    x = np.ascontiguousarray(x)
    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"./{model_name}.rbln")

    # Run inference
    rebel_result = module.run(x)

    # Display results
    print(f"Top1 category: {decode_predictions(rebel_result, top=1)[0][0][1]}")


if __name__ == "__main__":
    main()
