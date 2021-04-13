from efficientnet_pytorch import EfficientNet

from example import img

model = EfficientNet.from_pretrained('efficientnet-b0')

# ... image preprocessing as in the classification example ...
print(img.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(img)
print(features.shape) # torch.Size([1, 1280, 7, 7])


import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b1')
dummy_input = torch.randn(10, 3, 240, 240)

model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "test-b1.onnx", verbose=True)