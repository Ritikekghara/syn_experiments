from torchvision import datasets
from torch.utils.data import Dataset


class CytologyCombinedDataset(Dataset):
    def __init__(self, real_root, synth_root, transform=None, jsd_scores=None):
        self.real_ds = datasets.ImageFolder(real_root, transform=transform)
        self.synth_ds = datasets.ImageFolder(synth_root, transform=transform)

        # Sanity checks
        # ImageFolder sorts classes alphabetically, so ["BNV", "MEL"] is expected
        assert self.real_ds.classes == ["BNV", "MEL"], f"Expected classes ['BNV', 'MEL'], got {self.real_ds.classes}"
        assert self.synth_ds.classes == ["MEL"], f"Expected classes ['MEL'], got {self.synth_ds.classes}"  # only MEL synthetic images

        self.real_len = len(self.real_ds)
        self.synth_len = len(self.synth_ds)
        self.jsd_scores = jsd_scores or [0.0] * self.synth_len

        if len(self.jsd_scores) != self.synth_len:
            raise ValueError(
                f"JSD scores length ({len(self.jsd_scores)}) does not match synthetic dataset size ({self.synth_len})."
            )

    def __len__(self):
        return self.real_len + self.synth_len

    def __getitem__(self, idx):
        if idx < self.real_len:
            img, label = self.real_ds[idx]
            return img, label, False, 0.0
        else:
            img, _ = self.synth_ds[idx - self.real_len]
            jsd_score = float(self.jsd_scores[idx - self.real_len])
            return img, 1, True, jsd_score  # class index 1 = MEL (melanoma)
