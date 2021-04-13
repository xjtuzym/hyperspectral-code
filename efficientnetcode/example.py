import json
import time

from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0')

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
img = tfms(Image.open('1.png')).unsqueeze(0)

print(img.shape)  # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('C:\\Users\\85425\\PycharmProjects\\gaoguangpu\\images5\\labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1,2)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('-----')
cout = 0
for idx in torch.topk(outputs, k=2)[1].squeeze(0).tolist():
    cout+=1
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))
