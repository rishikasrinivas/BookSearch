import torch
import numpy as np 
class BalanceSampler(torch.utils.data.sampler.Sampler):
  def __init__ (self, data):
    self.data = data

    self.labels = torch.stack([self.data[entry_idx][2] for entry_idx in range(len(self.data))])
    self.sums = self.labels.sum(dim=0)
    self.avg = int(torch.mean(self.sums).item())


  def __len__(self):
    return len(self.data)

  def __iter__(self):
    training = []
    minority_classes = torch.where(self.sums < self.avg)[0]
    majority_classes = torch.where(self.sums >= self.avg)[0]

    for class_idx in minority_classes:
        class_indices = torch.where(self.labels[:, class_idx] == 1)[0]
        oversampled_indices = np.random.choice(class_indices, size=self.avg, replace=True)
        training.extend(oversampled_indices.tolist())

        # Undersample majority classes
    for class_idx in majority_classes:
        class_indices = torch.where(self.labels[:, class_idx] == 1)[0]
        undersampled_indices = np.random.choice(class_indices, size=self.avg, replace=False)
        training.extend(undersampled_indices.tolist())
    training=np.random.choice(training, size=6300, replace=False)


    return iter(training)

  def __getitem__(self, index):
        return self.data[index]
