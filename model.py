from torchvision.models import resnet18

model = resnet18(pretrained=True)
print(resnet18())
