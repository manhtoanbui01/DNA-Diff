import torch

val = -1.2305e-02
# Cast to float16
val_f16 = torch.tensor(val, dtype=torch.float16)

print(f"Original: {val}")
print(f"Float16:  {val_f16.item()}")