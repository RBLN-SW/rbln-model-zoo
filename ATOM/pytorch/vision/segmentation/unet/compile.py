import rebel  # RBLN Compiler
import torch


def main():
    model = torch.hub.load(
        "milesial/Pytorch-UNet", "unet_carvana", pretrained=False, scale=0.5
    )

    checkpoint_url = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth"

    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url, map_location=torch.device("cpu"), progress=True
    )

    model.load_state_dict(checkpoint)

    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, 320, 480], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    compiled_model.save("./unet.rbln")


if __name__ == "__main__":
    main()
