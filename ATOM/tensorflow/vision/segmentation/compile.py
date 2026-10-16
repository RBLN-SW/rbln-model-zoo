from pathlib import Path

import huggingface_hub as hf_hub
import rebel
import tensorflow as tf


def main():
    input_size = 512

    model_id = "keras-io/deeplabv3p-resnet50"

    def load_model():
        from_pretrained_keras = getattr(hf_hub, "from_pretrained_keras", None)
        if from_pretrained_keras is not None:
            model = from_pretrained_keras(model_id)
            return model

        # Manually load the model because from_pretrained_keras is unavailable on
        # huggingface-hub >= 1.0.0 when running under Python 3.9.
        snapshot_download = getattr(hf_hub, "snapshot_download", None)
        if snapshot_download is not None:
            local_dir = Path(snapshot_download(model_id, local_dir_use_symlinks=False))

            # Wrap the on-disk SavedModel so we can invoke it like a Keras layer.
            # This runs the exported signature directly instead of rebuilding the
            # weights as Python objects.
            tfsm_layer = tf.keras.layers.TFSMLayer(
                str(local_dir), call_endpoint="serving_default"
            )

            def model(input_img, layer=tfsm_layer):
                outputs = layer(input_img)
                return (
                    next(iter(outputs.values()))
                    if isinstance(outputs, dict)
                    else outputs
                )

            return model

        raise RuntimeError("Could not load the model.")

    model = load_model()

    func = tf.function(lambda input_img: model(input_img))

    input_info = [("input_img", [1, input_size, input_size, 3], tf.float32)]
    compiled_model = rebel.compile_from_tf_function(func, input_info)

    compiled_model.save("deeplabv3.rbln")


if __name__ == "__main__":
    main()
