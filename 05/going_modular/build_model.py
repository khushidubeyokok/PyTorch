'''Contains PyTorch code that initiates a TinyVGG model from the CNN explainer website'''
from torch import nn
class TinyVGG(nn.Module):
  def __init__(self,input,hidden_units,output):
    super().__init__()
    self.block1=nn.Sequential(
        nn.Conv2d(in_channels=input,out_channels=hidden_units,kernel_size=3,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block3=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.classifier=nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_features=hidden_units*4*4,out_features=output))

  def forward(self,x):
    x=self.block1(x)
    #print(x.shape)
    x=self.block2(x)
    #print(x.shape)
    x=self.block3(x)
    #print(x.shape)
    x=self.classifier(x)
    #print(x.shape)
    return x
