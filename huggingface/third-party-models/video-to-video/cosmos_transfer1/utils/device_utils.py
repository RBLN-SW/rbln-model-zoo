import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import rebel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cosmos-transfer1"))
from cosmos_transfer1.utils import log


class DEVICE_MAP:
    RBLN_DEVICE = {
        "text_encoder": {
            "device": "text_encoder/compiled_model.rbln",
        },
        "transformer": {
            "device": "transformer.rbln",
        },
        "vae": {
            "encoder": {
                "device": "vae/encoder.rbln",
            },
            "decoder": {
                "device": "vae/decoder.rbln",
            },
        },
        "safety_checker": {
            "qwen3guard": {
                "device": "safety_checker/qwen3guard",
            },
            # "siglip_encoder": {"device": "safety_checker/siglip_encoder"},
            # "video_safety_model": {"device": "safety_checker/video_safety_model"},
            "face_blur_filter": {
                "device": "safety_checker/face_blur_filter/retinaface.rbln"
            },
        },
        "prompt_upsampler": {
            "vision_tower": {
                "device": "prompt_upsampler/upsampler_model/vision_tower/compiled_model.rbln"
            },
            "language_model": {
                "device": "prompt_upsampler/upsampler_model/language_model",
            },
            "device": "prompt_upsampler/upsampler_model/compiled_model.rbln",
        },
        "preprocessor": {
            "sam2": "preprocessor/sam2",
            "depth_anything": "preprocessor/depth_anything/compiled_model.rbln",
            "grounding_dino": "preprocessor/grounding_dino",
        },
        "ctrlnet": {},
    }


@dataclass
class ModelInfo:
    """Model information with memory allocation per node"""

    name: str
    alloc_per_node: List[float]
    is_group: bool = False
    group_models: List[str] = None

    def __post_init__(self):
        if self.group_models is None:
            self.group_models = []

    @property
    def num_devices_required(self) -> int:
        return len(self.alloc_per_node)

    @property
    def max_memory_per_device(self) -> float:
        return max(self.alloc_per_node) if self.alloc_per_node else 0.0

    @property
    def display_name(self) -> str:
        """Display name for output (shows group info if applicable)"""
        if self.is_group and self.group_models:
            return f"{self.name} ({len(self.group_models)} models)"
        return self.name


class RBLNDeviceAllocator:
    """Optimal device allocation algorithm for RBLN models"""

    def __init__(
        self,
        device_memory_limit: float = 15.7,
        safety_margin: float = 0.5,
        max_devices: int = 16,
    ):
        self.device_memory_limit = device_memory_limit
        self.safety_margin = safety_margin  # Safety margin in GiB
        self.effective_memory_limit = (
            device_memory_limit - safety_margin
        )  # Actual usable memory
        self.max_devices = max_devices  # Maximum number of devices available
        self.models: List[ModelInfo] = []
        self.allocation_result: Dict[str, List[int]] = {}
        self.device_memory_usage: List[float] = [
            0.0
        ] * max_devices  # Pre-initialize with fixed size
        self.manual_allocations: Dict[
            str, List[int]
        ] = {}  # User-specified model-to-device mappings

    def extract_alloc_per_node(self, rbln_file: str) -> List[float]:
        """Extract alloc_per_node from a .rbln file"""

        try:
            metadata = rebel.RBLNCompiledModel.inspect(rbln_file)
            return [_mem / 1024**3 for _mem in metadata["alloc_per_node"]]
        except Exception as e:
            log.debug(f"Error inspecting {rbln_file}: {e}")
            return []

    def add_model_from_file(self, rbln_file: str, base_dir: str = ""):
        """Add a model from .rbln file"""
        alloc_per_node = self.extract_alloc_per_node(rbln_file)
        if alloc_per_node:
            name = os.path.relpath(rbln_file, base_dir) if base_dir else rbln_file
            model_info = ModelInfo(name, alloc_per_node)
            self.models.append(model_info)
            log.debug(f"Added model: {name}: {alloc_per_node} GiB")

    def scan_directory(
        self, rbln_dir: str, subfolders: List[str] = None, control_inputs: dict = None
    ):
        """Scan directory for .rbln files and add them"""
        # Add transformer.rbln if exists
        transformer_path = os.path.join(rbln_dir, "transformer.rbln")
        if os.path.exists(transformer_path):
            self.add_model_from_file(transformer_path, rbln_dir)

        if not subfolders:
            raise ValueError("subfolders list cannot be empty")

        preprocessor_map = {
            "depth": ["depth_anything"],
            "seg": ["sam2", "grounding_dino"],
        }

        # Required preprocessor models
        required_models = []
        if control_inputs:
            for hint_key in control_inputs:
                if hint_key in preprocessor_map:
                    required_models.extend(preprocessor_map[hint_key])

        # Scan subfolders
        for subfolder in subfolders:
            subfolder_path = os.path.join(rbln_dir, subfolder)
            rbln_files = glob.glob(
                os.path.join(subfolder_path, "**/*.rbln"), recursive=True
            )

            if not rbln_files:
                log.warning(f"No *.rbln files found in {subfolder}")

            for rbln_file in rbln_files:
                # Skip preprocessor files that are not required
                if subfolder == "preprocessor":
                    if any(hint_model in rbln_file for hint_model in required_models):
                        self.add_model_from_file(rbln_file, rbln_dir)
                elif subfolder == "ctrlnet":
                    if any(
                        hint_key + ".rbln" in rbln_file for hint_key in control_inputs
                    ):
                        self.add_model_from_file(rbln_file, rbln_dir)
                else:
                    self.add_model_from_file(rbln_file, rbln_dir)

    def _group_related_models(self):
        """Group models that should be allocated together based on their path patterns"""
        groups_to_create = {}

        # Group models by common path prefixes
        for model in self.models[:]:
            if model.is_group:
                continue

            # Extract potential group identifier
            path_parts = model.name.split("/")
            group_key = None

            # Check if it makes group with a specific model manually
            if (
                len(path_parts) >= 2
                and path_parts[0] == "preprocessor"
                and path_parts[1] == "sam2"
            ):
                group_key = f"{path_parts[0]}/{path_parts[1]}"

            elif (
                len(path_parts) >= 2
                and path_parts[0] == "preprocessor"
                and path_parts[1] == "grounding_dino"
            ):
                group_key = f"{path_parts[0]}/{path_parts[1]}"

            elif (
                len(path_parts) >= 2
                and path_parts[0] == "safety_checker"
                and "qwen3guard" == path_parts[1]
            ):
                group_key = f"{path_parts[0]}/{path_parts[1]}"

            # easy to model parsing
            elif (
                len(path_parts) >= 2
                and path_parts[0] == "safety_checker"
                and "video_content_safety_filter" == path_parts[1]
                and "siglip_encoder" in path_parts[2]
            ):
                group_key = f"{path_parts[0]}/{path_parts[2]}"

            # easy to model parsing
            elif (
                len(path_parts) >= 2
                and path_parts[0] == "safety_checker"
                and "video_content_safety_filter" == path_parts[1]
                and "safety_filter.rbln" in path_parts[2]
            ):
                group_key = f"{path_parts[0]}/video_safety_model"

            elif (
                len(path_parts) >= 2
                and path_parts[0] == "prompt_upsampler"
                and "upsampler_model" in path_parts[1]
                and "language_model" in path_parts[2]
            ):
                group_key = f"{path_parts[0]}/{path_parts[1]}/{path_parts[2]}"

            if group_key:
                if group_key not in groups_to_create:
                    groups_to_create[group_key] = []
                groups_to_create[group_key].append(model)

        # Create group models and remove individual models
        if groups_to_create:
            log.debug("\n=== Discovered model groups ===")

            for group_name, grouped_models in groups_to_create.items():
                # Check device requirements consistency across group
                all_same_device_count = (
                    len(set(model.num_devices_required for model in grouped_models))
                    == 1
                )

                # Multi-device group with consistent device count: sum per-device memory
                if all_same_device_count:
                    group_alloc = [0] * grouped_models[0].num_devices_required
                    for model in grouped_models:
                        group_alloc = [
                            x + y for x, y in zip(group_alloc, model.alloc_per_node)
                        ]
                else:
                    # Mixed device counts - skip for now as it's complex
                    log.debug(
                        f"Skipping mixed-device group '{group_name}' - "
                        "contains models with different device requirements"
                    )
                    continue

                # Create new group ModelInfo
                group_model = ModelInfo(
                    name=group_name,
                    alloc_per_node=group_alloc,
                    is_group=True,
                    group_models=[model.name for model in grouped_models],
                )

                log.debug(
                    f"Group '{group_name}': {len(grouped_models)} models, "
                    f"total memory: {group_alloc} GiB"
                )
                for model in grouped_models:
                    log.debug(f"  - {model.name}: {model.alloc_per_node} GiB")

                # Remove individual models from self.models and add group model
                for model in grouped_models:
                    self.models.remove(model)
                self.models.append(group_model)

    def can_fit_together(self, models: List[ModelInfo]) -> bool:
        """Check if multiple single-device models can fit in one device"""
        if any(model.num_devices_required > 1 for model in models):
            return False

        return (
            sum(model.alloc_per_node[0] for model in models)
            <= self.effective_memory_limit
        )

    def find_valid_device_range(
        self, model: "ModelInfo", max_device_index: int = None, start_from: int = 0
    ) -> int:
        """Find valid starting device index for multi-device models"""
        required_devices = model.num_devices_required

        if max_device_index is None:
            max_device_index = self.max_devices - 1

        # Check existing device ranges starting from start_from
        for start_idx in range(start_from, max_device_index + 1, required_devices):
            device_range = list(range(start_idx, start_idx + required_devices))
            # Ensure we don't exceed max devices
            if max(device_range) >= self.max_devices:
                break
            if self.can_allocate_to_range(model, device_range):
                return start_idx

        # If no existing range can accommodate, raise error
        self.print_device_memory_usage()
        raise ValueError(
            f"Cannot allocate {model.name} requiring {required_devices} devices. "
            f"No suitable device range found within {self.max_devices} available devices."
        )

    def can_allocate_to_range(
        self, model: "ModelInfo", device_range: List[int]
    ) -> bool:
        """Check if model can be allocated to the specified device range"""
        # Validate device range is within bounds
        if any(device_idx >= self.max_devices for device_idx in device_range):
            return False

        for i, device_idx in enumerate(device_range):
            # Get current memory usage for this device (direct indexing since we pre-initialized)
            current_memory = self.device_memory_usage[device_idx]

            # Get required memory for this device position
            required_memory = (
                model.alloc_per_node[i]
                if len(model.alloc_per_node) > 1
                else model.alloc_per_node[0]
            )

            # Check if this device has enough memory
            if current_memory + required_memory > self.effective_memory_limit:
                return False

        return True

    def _allocate_multi_device_models(
        self, result: Dict[str, List[int]], models_to_allocate: List[ModelInfo] = None
    ) -> None:
        """Allocate multi-device models (optionally from a specific list)"""
        if models_to_allocate is None:
            # Get multi-device models that are not already allocated in groups
            models_to_allocate = [
                m
                for m in self.models
                if m.num_devices_required > 1 and m.name not in result
            ]

        # Sort by maximum memory per device (descending), then by number of devices (ascending)
        # This ensures high-memory models get allocated first, allowing lower-memory models
        # to share device ranges
        models_to_allocate.sort(
            key=lambda x: (x.max_memory_per_device, -x.num_devices_required),
            reverse=True,
        )

        for model in models_to_allocate:
            required_devices = model.num_devices_required
            max_device_index = (
                len(self.device_memory_usage) - 1 if self.device_memory_usage else None
            )

            # For ctrlnet models, find non-overlapping device ranges
            if "ctrlnet" in model.name:
                start_idx = self._find_ctrlnet_device_range(model, result)
            else:
                start_idx = self.find_valid_device_range(model, max_device_index)

            # Create device list and allocation
            device_list = list(range(start_idx, start_idx + required_devices))
            success = self._allocate_model_to_devices(
                model, device_list, result, manual=False
            )

            if success:
                log.debug(
                    f"Multi-device allocation successful: {model.name} -> devices {device_list} (max_mem: {model.max_memory_per_device:.1f})"
                )
            else:
                self.print_device_memory_usage()
                error_msg = f"Multi-device allocation failed: {model.name} -> devices {device_list}. Please use more devices."
                raise ValueError(f"{error_msg}")

    def _allocate_single_device_models(
        self, result: Dict[str, List[int]], models_to_allocate: List[ModelInfo] = None
    ) -> None:
        """Greedy algorithm: prioritize existing device space first"""
        if models_to_allocate is None:
            # Get single device models that are not already allocated in groups
            models_to_allocate = [
                m
                for m in self.models
                if m.num_devices_required == 1 and m.name not in result
            ]

        models_to_allocate.sort(key=lambda x: x.alloc_per_node[0], reverse=True)
        remaining_models = models_to_allocate[:]

        while remaining_models:
            allocated_to_existing = False

            for model in remaining_models[:]:
                # Check existing devices for available space
                for i in range(len(self.device_memory_usage)):
                    current_device_memory = self.device_memory_usage[i]
                    # Check if device has enough space (both used devices and empty devices)
                    if (
                        model.alloc_per_node[0]
                        <= self.effective_memory_limit - current_device_memory
                    ):
                        success = self._allocate_model_to_devices(
                            model, [i], result, manual=False
                        )
                        if success:
                            remaining_models.remove(model)
                            allocation_type = (
                                "empty device"
                                if current_device_memory == 0.0
                                else "existing device"
                            )
                            log.debug(
                                f"Allocated device {i} to ['{model.name}'] (greedy - {allocation_type})"
                            )
                            allocated_to_existing = True
                            break

                if allocated_to_existing:
                    break

            # If no space found in any device, stop allocation
            if not allocated_to_existing and remaining_models:
                self.print_device_memory_usage()
                model = remaining_models[0]
                raise ValueError(
                    f"Cannot allocate {model.name} (requires {model.alloc_per_node[0]:.1f} GiB): "
                    f"No device has sufficient memory within {self.max_devices} device limit. "
                    f"Please use more devices or increase memory limit."
                )

    def allocate_devices(self) -> Dict[str, List[int]]:
        """Main allocation algorithm - minimize total device usage using greedy approach"""
        if not self.models:
            return {}

        result = {}

        log.debug("Starting model allocation")

        # discover and group related models
        self._group_related_models()

        # allocate manually specified models first
        self._allocate_manual_assignments(result)

        # allocate multi-device models
        self._allocate_multi_device_models(result)

        # allocate single-device models using greedy algorithm
        self._allocate_single_device_models(result)

        self.allocation_result = result
        return result

    def add_manual_allocations(self, allocations: Dict[str, List[int]]):
        """Add multiple manual allocations at once"""
        for model_name, device_ids in allocations.items():
            self.manual_allocations[model_name] = device_ids
            log.debug(f"Manual allocation set: {model_name} -> devices {device_ids}")

    def _allocate_manual_assignments(self, result: Dict[str, List[int]]) -> None:
        """
        Allocate models that have been manually assigned to specific devices
        This runs before automatic allocation algorithms
        """
        if not self.manual_allocations:
            return

        log.debug("=== Processing manual device assignments ===")

        for model_name, device_ids in self.manual_allocations.items():
            # Find the model in our models list
            model = None
            for m in self.models:
                if m.name == model_name:
                    model = m
                    break

            if model is None:
                log.warning(f"Manual allocation failed: Model '{model_name}' not found")
                continue

            # Validate device assignment
            required_devices = model.num_devices_required
            if len(device_ids) != required_devices:
                log.warning(
                    f"Manual allocation failed: Model '{model_name}' requires {required_devices} devices, "
                    f"but {len(device_ids)} devices specified: {device_ids}"
                )
                continue

            # Allocate to specific devices
            success = self._allocate_model_to_devices(
                model, device_ids, result, manual=True
            )

            if success:
                log.debug(
                    f"Manual allocation successful: {model.name} -> devices {device_ids}"
                )
            else:
                log.warning(
                    f"Manual allocation failed: Unable to allocate {model.name} to devices {device_ids}"
                )

    def _allocate_model_to_devices(
        self,
        model: "ModelInfo",
        device_ids: List[int],
        result: Dict[str, List[int]],
        manual: bool = False,
    ) -> bool:
        """Helper function to allocate a single model to specific devices"""
        # Validate device IDs are within bounds
        max_device = max(device_ids) if device_ids else -1
        if max_device >= self.max_devices:
            self.print_device_memory_usage()
            error_msg = (
                f"Device ID {max_device} exceeds maximum available devices ({self.max_devices}). "
                f"Cannot allocate {model.name} to devices {device_ids}."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Check if devices have enough memory (using direct indexing)
        for i, device_id in enumerate(device_ids):
            current_memory = self.device_memory_usage[device_id]
            required_memory = (
                model.alloc_per_node[i]
                if len(model.alloc_per_node) > 1
                else model.alloc_per_node[0]
            )

            if current_memory + required_memory > self.effective_memory_limit:
                error_msg = (
                    f"Device {device_id} insufficient memory for {model.name}: "
                    f"{current_memory:.1f} + {required_memory:.1f} > {self.effective_memory_limit:.1f} GiB"
                )
                if manual:
                    self.print_device_memory_usage()
                    log.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    log.debug(error_msg)
                    return False

        # Perform the allocation and update memory usage
        result[model.name] = device_ids.copy()
        memory_info = self._update_device_memory_usage(model, device_ids)

        allocation_type = "manual" if manual else "auto"
        log.debug(
            f"Allocated devices {device_ids} to {model.name} ({allocation_type}, {', '.join(memory_info)})"
        )

        return True

    def _update_device_memory_usage(
        self, model: ModelInfo, device_ids: List[int]
    ) -> List[str]:
        """Update device memory usage for a model allocation and return memory info strings"""
        memory_info = []

        for i, device_id in enumerate(device_ids):
            required_memory = (
                model.alloc_per_node[i]
                if len(model.alloc_per_node) > 1
                else model.alloc_per_node[0]
            )

            # Get current memory usage and update memory usage
            current_memory = self.device_memory_usage[device_id]
            self.device_memory_usage[device_id] += required_memory

            memory_info.append(
                f"D{device_id}: {current_memory:.1f}+{required_memory:.1f}={(current_memory + required_memory):.1f}"
            )

        return memory_info

    def _find_ctrlnet_device_range(
        self, model: "ModelInfo", result: Dict[str, List[int]]
    ) -> int:
        """Find non-overlapping device range for ControlNet models"""
        required_devices = model.num_devices_required

        # Collect all devices used by other ControlNet models
        ctrlnet_used_devices = set()
        for allocated_model_name, allocated_devices in result.items():
            if "ctrlnet" in allocated_model_name and allocated_model_name != model.name:
                ctrlnet_used_devices.update(allocated_devices)

        log.debug(f"ControlNet devices already used: {sorted(ctrlnet_used_devices)}")

        # Calculate maximum possible starting positions within device limits
        for start_idx in range(
            0, self.max_devices - required_devices + 1, required_devices
        ):
            # Generate device range for this starting position
            device_range = list(range(start_idx, start_idx + required_devices))

            # Check if this range overlaps with other ControlNet models
            if (
                not set(device_range).intersection(ctrlnet_used_devices)
            ) and self.can_allocate_to_range(model, device_range):
                log.debug(
                    f"Found non-overlapping range for {model.name}: devices {device_range}"
                )
                return start_idx

        # If no non-overlapping range found, raise error directly
        self.print_device_memory_usage()
        raise ValueError(
            f"Cannot allocate ControlNet model {model.name} requiring {required_devices} devices. "
            f"No non-overlapping device range available within {self.max_devices} devices. "
            f"Used ControlNet devices: {sorted(ctrlnet_used_devices)}"
        )

    def set_manual_allocation(self, model_name: str, device_ids: List[int]):
        """Set manual device allocation for a specific model"""
        self.manual_allocations[model_name] = device_ids
        log.debug(f"Manual allocation set: {model_name} -> devices {device_ids}")

    def print_device_memory_usage(self):
        """Print current device memory usage for debugging"""
        print("Current device memory usage:")
        for model in self.models:
            print(
                f" - {model.name}: {[f'{usage:.2f}' for usage in model.alloc_per_node]} GiB"
            )

    def print_allocation_summary(self):
        """Print allocation summary in a beautiful table format"""
        if not self.allocation_result:
            print("No allocation result available")
            return

        # Count actual unique devices used (not max device + 1)
        used_devices = set()
        for devices in self.allocation_result.values():
            used_devices.update(devices)
        total_devices = len(used_devices)

        # Calculate column width and table width early
        col_width = 25
        devices_per_row = 4
        table_width = 20 + (
            col_width * devices_per_row
        )  # 20 for metric column + device columns

        print(f"\n{'=' * table_width}")
        print(f"{'RBLN AUTO DEVICE ALLOCATOR SUMMARY':^{table_width}}")
        print(f"{'=' * table_width}")
        print(f"Total devices used: {total_devices}")
        print(f"Total models allocated: {len(self.allocation_result)}")
        print(f"Device memory limit: {self.device_memory_limit} GiB")
        print(
            f"Effective memory limit (with safety margin): {self.effective_memory_limit} GiB"
        )
        print(f"{'=' * table_width}")
        print("allocated memory per model")
        for model in self.models:
            print(
                f" - {model.name}: {[f'{usage:.2f}' for usage in model.alloc_per_node]} GiB -> Device {self.allocation_result.get(model.name, [])}"
            )
        print(f"{'=' * table_width}")

        # Group by device for summary and calculate memory usage
        device_usage = {}
        device_memory_usage = {}

        for model_name, devices in self.allocation_result.items():
            # Find the model info
            model_info = next((m for m in self.models if m.name == model_name), None)
            if model_info:
                for i, device in enumerate(devices):
                    if device not in device_usage:
                        device_usage[device] = []
                        device_memory_usage[device] = 0.0

                    # Add memory usage for this model on this device
                    if len(model_info.alloc_per_node) > 1:
                        device_memory_usage[device] += model_info.alloc_per_node[i]
                    else:
                        device_memory_usage[device] += model_info.alloc_per_node[0]

                    # Use display_name which handles group info automatically
                    device_usage[device].append(model_info.display_name)

        if not device_usage:
            print("No devices allocated")
            return

        # Create continuous device range from 0 to max used device
        used_device_ids = sorted(device_usage.keys())
        max_used_device = max(used_device_ids)
        continuous_devices = list(range(max_used_device + 1))

        # Group devices into chunks of 4
        device_chunks = [
            continuous_devices[i : i + devices_per_row]
            for i in range(0, len(continuous_devices), devices_per_row)
        ]

        # Print each chunk
        for chunk_idx, device_chunk in enumerate(device_chunks):
            if chunk_idx > 0:
                print()

            # Print table header
            print(f"{'-' * table_width}")
            header = f"{'Device ID':<20}"
            for device in device_chunk:
                header += f"{'Device ' + str(device):^{col_width}}"
            print(header)
            print(f"{'-' * table_width}")

            # Print allocated models (multi-line for better readability)
            def smart_truncate_model_name(name, max_len):
                """Simple truncation with ellipsis"""
                if len(name) <= max_len:
                    return name

                if max_len <= 2:
                    return name[:max_len]

                # Simple truncation with ".."
                return name[: max_len - 2] + ".."

            # Find max number of models in any device for this chunk
            max_models_in_chunk = max(
                len(device_usage.get(device, [])) for device in device_chunk
            )

            # Print each model line by line (starting with header row)
            for model_idx in range(max_models_in_chunk):
                if model_idx == 0:
                    model_line = f"{'Allocated Models':<20}"  # Header for first row
                else:
                    model_line = f"{'':<20}"

                for device in device_chunk:
                    models = device_usage.get(
                        device, []
                    )  # Empty list for unused devices

                    if model_idx < len(models):
                        # Calculate available space for model name (leave some margin)
                        max_name_len = col_width - 4
                        model_name = smart_truncate_model_name(
                            models[model_idx], max_name_len
                        )
                        model_str = f"'{model_name}'"
                    else:
                        model_str = ""

                    model_line += f"{model_str:^{col_width}}"

                print(model_line)

            # Print used memory
            used_memory_row = f"{'Used Memory (GiB)':<20}"
            for device in device_chunk:
                used_memory = device_memory_usage.get(device, 0.0)
                used_memory_row += f"{used_memory:.2f}".center(col_width)
            print(used_memory_row)

            # Print remaining memory
            remaining_memory_row = f"{'Remaining (GiB)':<20}"
            for device in device_chunk:
                used_memory = device_memory_usage.get(device, 0.0)
                remaining_memory = self.device_memory_limit - used_memory
                remaining_memory_row += f"{remaining_memory:.2f}".center(col_width)
            print(remaining_memory_row)

            # Print utilization
            utilization_row = f"{'Utilization (%)':<20}"
            for device in device_chunk:
                used_memory = device_memory_usage.get(device, 0.0)
                utilization = (used_memory / self.device_memory_limit) * 100
                utilization_row += f"{utilization:.1f}%".center(col_width)
            print(utilization_row)

            # Print memory bar visualization
            print(f"{'-' * table_width}")
            memory_bar_row = f"{'Memory Usage':<20}"
            for device in device_chunk:
                used_memory = device_memory_usage.get(device, 0.0)
                utilization = used_memory / self.device_memory_limit
                bar_length = 12  # Fixed bar length for consistency
                filled_length = int(bar_length * utilization)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                memory_bar_row += f"[{bar}]".center(col_width)
            print(memory_bar_row)

            print(f"{'-' * table_width}")

        print()


def get_rbln_device(
    rbln_dir, control_inputs, subfolders=None, print_summary=True, max_devices=16
):
    allocator = RBLNDeviceAllocator(
        device_memory_limit=15.7, safety_margin=0.2, max_devices=max_devices
    )
    if subfolders is None:
        subfolders = [
            "text_encoder",
            "vae",
            "ctrlnet",
            "prompt_upsampler",
            "safety_checker",
            "preprocessor",
        ]
    log.debug("=== Scanning for .rbln files ===")
    allocator.scan_directory(rbln_dir, subfolders, control_inputs)

    # Set manual allocations for transformer and text encoder if they can fit together
    transformer_mem = 15.7
    text_encoder_mem = 15.7
    vae_encoder_mem = 15.7
    for model in allocator.models:
        if model.name == "transformer.rbln":
            transformer_mem = model.alloc_per_node[-1]
            transformer_devices = len(model.alloc_per_node)
        if model.name == "text_encoder/compiled_model.rbln":
            text_encoder_mem = model.alloc_per_node[-1]
        if model.name == "vae/encoder.rbln":
            vae_encoder_mem = model.alloc_per_node[-1]

    if text_encoder_mem < vae_encoder_mem:
        if (
            transformer_mem + vae_encoder_mem <= allocator.effective_memory_limit
        ):  # FIXME : temp patch
            allocator.add_manual_allocations(
                {
                    "transformer.rbln": [i for i in range(transformer_devices)],
                    "vae/encoder.rbln": [1],  # hard coded
                }
            )
    else:
        if transformer_mem + text_encoder_mem <= allocator.effective_memory_limit:
            allocator.add_manual_allocations(
                {
                    "transformer.rbln": [i for i in range(transformer_devices)],
                    "text_encoder/compiled_model.rbln": [1],  # hard coded
                }
            )

    log.debug(f"\n=== Found {len(allocator.models)} models ===")

    log.debug("\n=== Starting device allocation ===")
    allocation_result = allocator.allocate_devices()

    if print_summary:
        allocator.print_allocation_summary()

    rbln_device_config = parse_config(
        DEVICE_MAP.RBLN_DEVICE, allocation_result, subfolders
    )

    return rbln_device_config


def parse_config(device_config, allocation_result, subfolders):
    device_config["transformer"].update(
        {
            "device": get_device_from_result(
                allocation_result, device_config["transformer"]["device"]
            )
        }
    )
    for subfolder in subfolders:
        if subfolder == "vae":
            for part in ["encoder", "decoder"]:
                device_config[subfolder][part].update(
                    {
                        "device": get_device_from_result(
                            allocation_result, device_config[subfolder][part]["device"]
                        )
                    }
                )
        elif subfolder in "safety_checker":
            for part in device_config[subfolder].keys():
                device_config[subfolder][part].update(
                    {
                        "device": get_device_from_result(
                            allocation_result, device_config[subfolder][part]["device"]
                        )
                    }
                )
        elif subfolder == "preprocessor":
            for part in device_config[subfolder].keys():
                device_config[subfolder].update(
                    {
                        part: get_device_from_result(
                            allocation_result, device_config[subfolder][part]
                        )
                    }
                )
        elif subfolder == "prompt_upsampler":
            for part in ["vision_tower", "language_model"]:
                device_config[subfolder][part].update(
                    {
                        "device": get_device_from_result(
                            allocation_result, device_config[subfolder][part]["device"]
                        )
                    }
                )
            device_config[subfolder].update(
                {
                    "device": get_device_from_result(
                        allocation_result, device_config[subfolder]["device"]
                    )
                }
            )
        elif subfolder == "ctrlnet":
            for model_path, device in allocation_result.items():
                if "ctrlnet" in model_path:
                    path_parts = model_path.split("/")
                    device_config[subfolder].update(
                        {path_parts[1][:-5]: {"device": device}}
                    )
        else:
            device_config[subfolder].update(
                {
                    "device": get_device_from_result(
                        allocation_result, device_config[subfolder]["device"]
                    )
                }
            )

    return device_config


def get_device_from_result(result, filename):
    try:
        allocated_device = result[filename]
    except:  # noqa: E722
        log.warning(
            f"Device location {filename} not found in allocation result. "
            f"It will be allocated to device -1."
        )
        allocated_device = -1
    return allocated_device
