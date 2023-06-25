import torch


class ManualLoader:
    """
    Create the data loader with a manual list as the sequence
    """

    def __init__(
        self,
        ae_sequence: torch.Tensor,
        lot_size_sequence: torch.Tensor,
        directory: str,
        sequence_name: str,
        ae_type: str,
    ):
        self.sequences: list[torch.Tensor] = [
            torch.cat([ae_sequence.unsqueeze(1), lot_size_sequence.unsqueeze(1)], dim=1)
        ]
        self.metadata: dict = {"dir": directory, "sequence_name": sequence_name, "ae_type": ae_type}
