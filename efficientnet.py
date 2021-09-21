import torch
import torch.nn as nn
import torch.nn.functional as F


class SEmodule(nn.Module):
    def __init__(self,in_channel,reduction=4):
        super(SEmodule, self).__init__()
        self.linear1 = nn.Linear(in_channel,in_channel//reduction)
        self.act1 = nn.SiLU()
        self.linear2 = nn.Linear(in_channel//reduction,in_channel)
    def forward(self,x):
        skip = x
        b,c,_,_ = x.shape
        x = self.linear2(self.act1(self.linear1(x.mean(dim=(2,3)))))
        x = torch.sigmoid(x.view(b,c,1,1)) * skip
        return x

class MBConv(nn.Module):
    def __init__(self,in_features,out_features,expand,kernel=3,stride=1):
        super(MBConv, self).__init__()
        self.use_skip = (in_features==out_features) and (stride == 1)
        self.conv1 = nn.Conv2d(in_features,in_features*expand,1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_features*expand)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_features*expand,in_features*expand,kernel,stride=stride,padding=kernel//2,
                               groups=in_features*expand,bias=False)
        self.bn2 = nn.BatchNorm2d(in_features*expand)
        self.act2 = nn.SiLU()
        self.se = SEmodule(in_features*expand)
        self.conv3 = nn.Conv2d(in_features*expand,out_features,1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self,x):
        skip = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = self.bn3(self.conv3(x))
        if self.use_skip:
            x = skip+x
        return x


class MBconvLayers(nn.Module):
    def __init__(self,in_channel,out_channel,expand,kernel,stride,layers):
        super(MBconvLayers, self).__init__()
        layer = [MBConv(in_channel,out_channel,expand,kernel,stride)]
        for i in range(layers-1):
            layer.append(MBConv(out_channel,out_channel,expand,kernel,1))
        self.main = nn.Sequential(*layer)
    def forward(self,x):
        return self.main(x)



class EfficientNet(nn.Module):
    def __init__(self,in_channel,out_class):
        super(EfficientNet, self).__init__()
        # [expand, kernel, stride, channel, layers]
        setting = [
            [16,1,3,1,1],
            [24,6,3,2,2],
            [40,6,5,2,2],
            [80,6,3,2,3],
            [112,6,5,1,3],
            [192,6,5,2,4],
            [320,6,3,1,1],
        ]
        layer = []
        in_ch = 32
        for i in setting:
            layer.append(MBconvLayers(in_ch,*i))
            in_ch = i[0]
        self.main = nn.Sequential(*layer)
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channel,32,kernel_size=3,padding=1,stride=2,bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.last_conv = nn.Conv2d(320,1280,1,1)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(1280,out_class)

    def forward(self,x):
        x = self.pre_conv(x)
        x = self.main(x)
        x = self.last_conv(x)
        x = torch.mean(x,dim=(2,3))
        x = self.linear(self.dropout(x))
        return torch.softmax(x,dim=-1)

if __name__ == '__main__':
    net = EfficientNet(3,10)
    inp = torch.randn((1,3,224,224),requires_grad=True)
    out = net(inp)
    torch.sum(out).backward()
    print(inp.grad)
