# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import types
from pathlib import Path


def _load_vllm_utils(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class DummyTorchType:
        pass

    fake_torch.Tensor = DummyTorchType
    fake_torch.Size = DummyTorchType
    fake_torch.dtype = DummyTorchType
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "zmq", types.ModuleType("zmq"))

    fake_device = types.ModuleType("verl.utils.device")
    fake_device.is_npu_available = False
    fake_device.get_torch_device = lambda: None
    monkeypatch.setitem(sys.modules, "verl.utils.device", fake_device)

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")
    fake_vllm_utils.TensorLoRARequest = DummyTorchType

    class DummyHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.VLLMHijack = DummyHijack
    monkeypatch.setitem(sys.modules, "verl.utils.vllm", fake_vllm_utils)

    fake_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_patch.patch_vllm_moe_model_weight_loader = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.patch", fake_patch)

    fake_fp8_utils = types.ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fake_fp8_utils.apply_vllm_fp8_patches = lambda *args, **kwargs: None
    fake_fp8_utils.is_fp8_model = lambda *args, **kwargs: False
    fake_fp8_utils.load_quanted_weights = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.vllm_fp8_utils", fake_fp8_utils)

    fake_verl = types.ModuleType("verl")
    fake_utils = types.ModuleType("verl.utils")
    fake_workers = types.ModuleType("verl.workers")
    fake_rollout = types.ModuleType("verl.workers.rollout")
    fake_vllm_rollout = types.ModuleType("verl.workers.rollout.vllm_rollout")
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "verl.workers", fake_workers)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout", fake_rollout)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout", fake_vllm_rollout)

    utils_path = (
        Path(__file__).resolve().parents[3] / "verl" / "workers" / "rollout" / "vllm_rollout" / "utils.py"
    )
    spec = importlib.util.spec_from_file_location("verl.workers.rollout.vllm_rollout.utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_get_device_uuid_fallback(monkeypatch):
    fake_platforms = types.ModuleType("vllm.platforms")

    class DummyPlatform:
        def get_device_uuid(self, device_id):
            raise NotImplementedError

    fake_platforms.current_platform = DummyPlatform()
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.platforms = fake_platforms

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    utils = _load_vllm_utils(monkeypatch)

    assert utils.get_device_uuid(1) == "CUDA-2-3-1"


def test_get_device_uuid_fallback_without_visible_devices(monkeypatch):
    fake_platforms = types.ModuleType("vllm.platforms")

    class DummyPlatform:
        def get_device_uuid(self, device_id):
            raise NotImplementedError

    fake_platforms.current_platform = DummyPlatform()
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.platforms = fake_platforms

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    utils = _load_vllm_utils(monkeypatch)

    assert utils.get_device_uuid(1) == "CUDA-1-1"
