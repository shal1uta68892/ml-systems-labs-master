import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab06.src.run_lab06 import make_model, train_one_epoch


class TrainingPipelineTests(unittest.TestCase):
    def test_lab3_cnn_shape_and_parameter_count(self) -> None:
        model = make_model()
        output = model(torch.zeros(2, 1, 28, 28))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 20490)

    def test_accumulation_matches_effective_batch_update(self) -> None:
        torch.manual_seed(7)
        full_model = nn.Linear(4, 3)
        accumulated_model = nn.Linear(4, 3)
        accumulated_model.load_state_dict(full_model.state_dict())
        inputs = torch.randn(256, 4)
        targets = torch.randint(0, 3, (256,))
        dataset = TensorDataset(inputs, targets)
        full_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        micro_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
        accumulated_optimizer = torch.optim.SGD(
            accumulated_model.parameters(), lr=0.1
        )

        train_one_epoch(
            full_model, full_loader, full_optimizer, torch.device("cpu")
        )
        train_one_epoch(
            accumulated_model,
            micro_loader,
            accumulated_optimizer,
            torch.device("cpu"),
            accumulation_steps=4,
        )

        for full_parameter, accumulated_parameter in zip(
            full_model.parameters(), accumulated_model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                full_parameter, accumulated_parameter, rtol=1e-5, atol=1e-7
            )


if __name__ == "__main__":
    unittest.main()
