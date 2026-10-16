import argparse

import rebel  # RBLN Compiler
import tensorflow as tf
import tf_keras.applications as app

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

    # Instantiate TF Keras model
    model = getattr(app, model_name)(weights="imagenet")

    # TODO(@RBLN): remove `tf.config.run_functions_eagerly()` when
    # `grouped convolutions` are fully supported on the CPU by Keras
    tf.config.run_functions_eagerly(True)

    func = tf.function(lambda input_img: model(input_img))

    input_size = input_size_map.get(model_name, 224)

    # Compile TF Keras model for ATOM
    input_info = [("input_img", [1, input_size, input_size, 3], tf.float32)]
    compiled_model = rebel.compile_from_tf_function(func, input_info)
    tf.config.run_functions_eagerly(False)

    # Save compiled results to disk
    compiled_model.save(f"./{model_name}.rbln")


if __name__ == "__main__":
    main()
