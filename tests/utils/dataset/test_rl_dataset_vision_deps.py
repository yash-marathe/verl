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

import pytest


def test_missing_vision_deps_disables_processor(monkeypatch, caplog):
    rl_dataset = _load_rl_dataset(monkeypatch)
    caplog.set_level("WARNING")

    class DummyConfig:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key, default)

    class DummyDataset:
        def __len__(self):
            return 0

        def filter(self, func, num_proc=None, desc=None):
            return self

    def fake_load_dataset(*args, **kwargs):
        return {"train": DummyDataset()}

    monkeypatch.setattr(rl_dataset.RLHFDataset, "_download", lambda self, use_origin_parquet=False: None)
    monkeypatch.setattr(rl_dataset.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(rl_dataset.datasets, "concatenate_datasets", lambda frames: frames[0])

    original_import = rl_dataset.importlib.import_module

    def fake_import_module(name, package=None):
        if name == "verl.utils.dataset.vision_utils":
            raise ImportError("missing vision deps")
        return original_import(name, package)

    monkeypatch.setattr(rl_dataset.importlib, "import_module", fake_import_module)

    class DummyTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
            return [1, 2, 3]

    class DummyProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return {"input_ids": [[1, 2]]}

    dataset = rl_dataset.RLHFDataset(
        data_files="dummy.parquet",
        tokenizer=DummyTokenizer(),
        config=DummyConfig(
            {
                "cache_dir": "/tmp",
                "prompt_key": "prompt",
                "image_key": "images",
                "video_key": "videos",
                "image_patch_size": 14,
                "max_prompt_length": 1024,
                "return_raw_chat": False,
                "return_full_prompt": False,
                "truncation": "error",
                "filter_overlong_prompts": True,
                "apply_chat_template_kwargs": {},
                "tool_config_path": None,
                "filter_overlong_prompts_workers": 1,
                "use_shm": False,
                "chat_template_func": None,
                "need_tools_kwargs": False,
                "filter_prompts": True,
                "return_multi_modal_inputs": True,
                "shuffle": False,
                "seed": None,
                "ignore_missing_vision_deps": True,
            }
        ),
        processor=DummyProcessor(),
    )

    assert dataset.processor is None
    assert any("Skipping multimodal prompt filtering" in record.message for record in caplog.records)


def test_missing_vision_deps_raises_when_not_ignored(monkeypatch):
    rl_dataset = _load_rl_dataset(monkeypatch)

    class DummyConfig:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key, default)

    class DummyDataset:
        def __len__(self):
            return 0

        def filter(self, func, num_proc=None, desc=None):
            return self

    def fake_load_dataset(*args, **kwargs):
        return {"train": DummyDataset()}

    monkeypatch.setattr(rl_dataset.RLHFDataset, "_download", lambda self, use_origin_parquet=False: None)
    monkeypatch.setattr(rl_dataset.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(rl_dataset.datasets, "concatenate_datasets", lambda frames: frames[0])

    original_import = rl_dataset.importlib.import_module

    def fake_import_module(name, package=None):
        if name == "verl.utils.dataset.vision_utils":
            raise ImportError("missing vision deps")
        return original_import(name, package)

    monkeypatch.setattr(rl_dataset.importlib, "import_module", fake_import_module)

    class DummyTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
            return [1, 2, 3]

    class DummyProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return {"input_ids": [[1, 2]]}

    with pytest.raises(ImportError):
        rl_dataset.RLHFDataset(
            data_files="dummy.parquet",
            tokenizer=DummyTokenizer(),
            config=DummyConfig(
                {
                    "cache_dir": "/tmp",
                    "prompt_key": "prompt",
                    "image_key": "images",
                    "video_key": "videos",
                    "image_patch_size": 14,
                    "max_prompt_length": 1024,
                    "return_raw_chat": False,
                    "return_full_prompt": False,
                    "truncation": "error",
                    "filter_overlong_prompts": True,
                    "apply_chat_template_kwargs": {},
                    "tool_config_path": None,
                    "filter_overlong_prompts_workers": 1,
                    "use_shm": False,
                    "chat_template_func": None,
                    "need_tools_kwargs": False,
                    "filter_prompts": True,
                    "return_multi_modal_inputs": True,
                    "shuffle": False,
                    "seed": None,
                    "ignore_missing_vision_deps": False,
                }
            ),
            processor=DummyProcessor(),
        )


def _load_rl_dataset(monkeypatch):
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *args, **kwargs: {"train": []}
    fake_datasets.concatenate_datasets = lambda frames: frames[0]

    class DummyDatasetType:
        pass

    fake_datasets.Dataset = DummyDatasetType
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.fromiter = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    fake_torch = types.ModuleType("torch")

    class DummyTensor:
        pass

    fake_torch.Tensor = DummyTensor
    fake_torch.tensor = lambda *args, **kwargs: DummyTensor()
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")

    class DummyDatasetBase:
        pass

    fake_torch_utils_data.Dataset = DummyDatasetBase
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")

    class DummyImage:
        pass

    fake_pil_image.Image = DummyImage
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)

    fake_omegaconf = types.ModuleType("omegaconf")

    class DummyDictConfig(dict):
        pass

    class DummyListConfig(list):
        pass

    fake_omegaconf.DictConfig = DummyDictConfig
    fake_omegaconf.ListConfig = DummyListConfig
    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)

    fake_transformers = types.ModuleType("transformers")

    class DummyTokenizer:
        pass

    class DummyProcessorMixin:
        pass

    fake_transformers.PreTrainedTokenizer = DummyTokenizer
    fake_transformers.ProcessorMixin = DummyProcessorMixin
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_verl = types.ModuleType("verl")
    fake_utils = types.ModuleType("verl.utils")
    fake_dataset_pkg = types.ModuleType("verl.utils.dataset")
    fake_import_utils = types.ModuleType("verl.utils.import_utils")
    fake_import_utils.load_extern_object = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "verl.utils.dataset", fake_dataset_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils.import_utils", fake_import_utils)

    rl_dataset_path = Path(__file__).resolve().parents[3] / "verl" / "utils" / "dataset" / "rl_dataset.py"
    spec = importlib.util.spec_from_file_location("verl.utils.dataset.rl_dataset", rl_dataset_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
