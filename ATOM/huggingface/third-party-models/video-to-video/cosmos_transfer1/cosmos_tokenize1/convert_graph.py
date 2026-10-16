import functools

import torch

VAE_DECODER_PATH = [
    "post_quant_conv",
    "decoder.conv_in",
    "decoder.conv_out",
    "decoder.mid.block_1.conv1",
    "decoder.mid.block_1.conv2",
    "decoder.mid.block_2.conv1",
    "decoder.mid.block_2.conv2",
    "decoder.mid.attn_1.0.q",
    "decoder.mid.attn_1.0.k",
    "decoder.mid.attn_1.0.v",
    "decoder.mid.attn_1.0.proj_out",
    "decoder.mid.attn_1.1.q",
    "decoder.mid.attn_1.1.k",
    "decoder.mid.attn_1.1.v",
    "decoder.mid.attn_1.1.proj_out",
    "decoder.up.0.block.0.conv1",
    "decoder.up.0.block.0.conv2",
    "decoder.up.0.block.0.nin_shortcut",
    "decoder.up.0.block.1.conv1",
    "decoder.up.0.block.1.conv2",
    "decoder.up.0.block.2.conv1",
    "decoder.up.0.block.2.conv2",
    "decoder.up.1.block.0.conv1",
    "decoder.up.1.block.0.conv2",
    "decoder.up.1.block.1.conv1",
    "decoder.up.1.block.1.conv2",
    "decoder.up.1.block.2.conv1",
    "decoder.up.1.block.2.conv2",
    "decoder.up.1.upsample.conv1",
    "decoder.up.1.upsample.conv2",
    "decoder.up.1.upsample.conv3",
    "decoder.up.2.block.0.conv1",
    "decoder.up.2.block.0.conv2",
    "decoder.up.2.block.1.conv1",
    "decoder.up.2.block.1.conv2",
    "decoder.up.2.block.2.conv1",
    "decoder.up.2.block.2.conv2",
]

VAE_ENCODER_PATH = [
    "quant_conv",
    "encoder.conv_in",
    "encoder.conv_out",
    "encoder.mid.block_1.conv1",
    "encoder.mid.block_1.conv2",
    "encoder.mid.block_2.conv1",
    "encoder.mid.block_2.conv2",
    "encoder.mid.attn_1.0.q",
    "encoder.mid.attn_1.0.k",
    "encoder.mid.attn_1.0.v",
    "encoder.mid.attn_1.0.proj_out",
    "encoder.mid.attn_1.1.q",
    "encoder.mid.attn_1.1.k",
    "encoder.mid.attn_1.1.v",
    "encoder.mid.attn_1.1.proj_out",
    "encoder.down.0.block.0.conv1",
    "encoder.down.0.block.0.conv2",
    "encoder.down.0.block.0.nin_shortcut",
    "encoder.down.0.block.1.conv1",
    "encoder.down.0.block.1.conv2",
    "encoder.down.0.downsample.conv1",
    "encoder.down.0.downsample.conv3",
    "encoder.down.1.block.0.conv1",
    "encoder.down.1.block.0.conv2",
    "encoder.down.1.block.0.nin_shortcut",
    "encoder.down.1.block.1.conv1",
    "encoder.down.1.block.1.conv2",
    "encoder.down.2.block.0.conv1",
    "encoder.down.2.block.0.conv2",
    "encoder.down.2.block.1.conv1",
    "encoder.down.2.block.1.conv2",
]


def optimize_module_by_path(base_module, path_str):
    parts = path_str.split(".")
    target_module = functools.reduce(getattr, parts, base_module)

    if hasattr(target_module, "0"):
        original_child = getattr(target_module, "0")
        optimized_child = optimize_graph_from_constants(original_child)
        setattr(target_module, "0", optimized_child)
    else:
        original_child = target_module
        optimized_child = optimize_graph_from_constants(original_child)
        target_module = optimized_child


def optimize_graph_from_constants(module: torch.jit.ScriptModule):
    graph = module.graph
    try:
        repeat_node = next(
            node for node in graph.nodes() if node.kind() == "aten::repeat"
        )
        repeats_list_value = list(repeat_node.inputs())[1]
        repeats_list_construct_node = repeats_list_value.node()

        if repeats_list_construct_node.kind() != "prim::ListConstruct":
            raise StopIteration("Could not find 'prim::ListConstruct' for repeat dims.")

        temporal_repeat_value = list(repeats_list_construct_node.inputs())[2]
        temporal_repeat_node = temporal_repeat_value.node()

        if (
            temporal_repeat_node.kind() == "prim::Constant"
            and temporal_repeat_node.i("value") == 0
        ):
            pass
        else:
            return module

    except (StopIteration, IndexError, RuntimeError):
        return module

    try:
        tensor_list_construct_node = repeat_node.output().uses()[0].user
        cat_node = tensor_list_construct_node.output().uses()[0].user
        node_after_padding = cat_node.output().uses()[0].user

        slice_3_node = list(repeat_node.inputs())[0].node()
        slice_2_node = list(slice_3_node.inputs())[0].node()
        slice_1_node = list(slice_2_node.inputs())[0].node()

        original_input_value = list(slice_1_node.inputs())[0]

        nodes_to_remove = [
            cat_node,
            tensor_list_construct_node,
            repeat_node,
            repeats_list_construct_node,
            slice_3_node,
            slice_2_node,
            slice_1_node,
        ]

    except (StopIteration, IndexError):
        return None

    node_after_padding.replaceInputWith(cat_node.output(), original_input_value)
    for node in nodes_to_remove:
        node.destroy()

    try:
        graph.lint()
        return module
    except Exception:
        return None
