from datasets import load_from_disk
import torch

dataset = load_from_disk("/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_with_ipadapter/")

print(dataset)
print(len(dataset))
data = dataset[0:100]
print(data['label'])

lists = dataset['cond_embed']
print(torch.equal(torch.Tensor(lists[0]), torch.Tensor(lists[10])))
