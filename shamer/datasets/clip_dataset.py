from __future__ import annotations
import torch
import torchcodec


class ClipDataset(torch.utils.data.Dataset):

    def __init__(
        self: ClipDataset,
        samples,
    ) -> None:

        super(ClipDataset, self).__init__()
        self.samples = samples

    def __len__(self: ClipDataset) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self: ClipDataset, index: int) -> tuple[torch.Tensor, int]:
        """Returns the sample at the given index."""
        sample = self.samples[index]
        video_path = sample["path"]
        indices = sample["indices"]

        video_decoder = torchcodec.decoders.VideoDecoder(
            video_path)  # type: ignore
        frames = video_decoder.get_frames_at(indices)  # type: ignore

        class_label = sample["class"]

        return frames.data, class_label

    @property
    def samples(self: ClipDataset):
        return self._samples

    @samples.setter
    def samples(
        self: ClipDataset,
        samples
    ) -> None:
        """Sets the samples of the dataset."""
        samples = sorted(samples, key=lambda sample: (
            sample["path"], sample["indices"]))
        self._samples = samples

    def filter_classes(
        self: ClipDataset,
        class_filter: set[int]
    ) -> None:
        """Filters the dataset to only include samples with classes in the given set."""
        self._samples = [
            sample for sample in self._samples if sample["class"] in class_filter]

    def remap_classes(
        self: ClipDataset,
        class_mapping: dict[int, int]
    ) -> None:
        for sample in self._samples:
            sample["class"] = class_mapping[sample["class"]]  # type: ignore
