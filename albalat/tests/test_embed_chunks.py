import torch
from albalat.scripts.embed_chunks import compute_start_end_indexes, load_model



class TestEmbedChunks:
    def test_compute_start_end_indexes(self):
        local_rank0 = 0
        local_rank1 = 1
        local_rank2 = 2
        world_size = 3
        n_chunks = 39874
        indexes_0 = compute_start_end_indexes(local_rank0, world_size, n_chunks)
        indexes_1 = compute_start_end_indexes(local_rank1, world_size, n_chunks)
        indexes_2 = compute_start_end_indexes(local_rank2, world_size, n_chunks)
        assert indexes_0 == (0, 13291) and indexes_1 == (13291, 26582) and indexes_2 == (26582, 39875) , f"""
                                            Expected indexes (0, 13291), (13292, 26582),(26582, 39875) but got
                                            {indexes_0}, {indexes_1}, {indexes_2}."""

    def test_load_model_wrong_dtype(self):
        has_raised = False
        try:
            load_model("IEITYuan/Yuan-embedding-2.0-en", "cpu", "torch.bfloat654",
                   "spda")

        except AssertionError:
            has_raised = True

        assert has_raised, "The wrong input type should have raised an exception."

def test_load_model_device_cpu():
    model = load_model("IEITYuan/Yuan-embedding-2.0-en", "cpu", "torch.bfloat654",
               "spda")
    assert model.device == "cpu", f"""The model should be on cpu, current {model.device}"""

def test_load_model_device_gpu():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.mps.is_available():
        device = "mps:0"
    else:
        device = "cpu"

    model = load_model("IEITYuan/Yuan-embedding-2.0-en", device, "torch.bfloat16",
               "sdpa")

    assert str(model.device) == device, f"""The model should be on cpu, current {model.device}"""

def test_load_model_dtype():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.mps.is_available():
        device = "mps:0"
    else:
        device = "cpu"

    model = load_model("IEITYuan/Yuan-embedding-2.0-en", device, "torch.bfloat16",
               "sdpa")
    assert next(model.parameters()).dtype == torch.bfloat16, f"""The model should be on cpu, current {model.device}"""