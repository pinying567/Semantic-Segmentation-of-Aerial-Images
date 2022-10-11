import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from vgg import make_layers, cfgs, model_urls

class VGG_FCN32s(nn.Module):
    def __init__(self, features, num_classes=7):
        super(VGG_FCN32s, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.features[0].padding = (100, 100)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1),
            nn.ConvTranspose2d(num_classes, num_classes, 64, 32, bias=False)
        )
        self._initialize_weights()
        
    def forward(self, x):
        # x: Bx3x224x224
        x = self.features(x)                       # Bx512x13x13
        x = self.decoder(x)[:, :, 16:-16, 16:-16]  # BxCx224x224
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if state_dict[name].size() != own_state[name].size():
                    print('Skip loading parameter {}'.format(name))
                    continue
                own_state[name].copy_(param)

def _vgg_fcn32s(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG_FCN32s(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        print('Loading pretrained model ...')
        state_dict = model_zoo.load_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16_fcn32s(pretrained=True, progress=True, **kwargs):
    return _vgg_fcn32s('vgg16', 'D', False, pretrained, progress, **kwargs)

"""
import pdb
batch_size = 4
x = torch.randn([batch_size, 3, 224, 224], dtype=torch.float).cuda()
model = vgg16_fcn32s(**{'num_classes': 7}).cuda()
output = model(x)
prob, pred = torch.max(torch.softmax(output, dim=1), dim=1)
pdb.set_trace()
"""
