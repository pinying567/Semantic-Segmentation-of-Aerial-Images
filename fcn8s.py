import functools
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from vgg import make_layers, cfgs, model_urls
   
class VGG_FCN8s(nn.Module):
    def __init__(self, features, num_classes=7):
        super(VGG_FCN8s, self).__init__()
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
            nn.Conv2d(4096, num_classes, 1)
        )
        self.decoder_pool4 = nn.Conv2d(512, num_classes, 1)
        self.decoder_pool3 = nn.Conv2d(256, num_classes, 1)
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, bias=False)
        self.upsample4 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, bias=False)
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, bias=False)
        self._initialize_weights()
        
    def forward(self, x):
        # x: Bx3x224x224
        # extract features
        conv3 = self.features[0:17](x)        # Bx256x52x52
        conv4 = self.features[17:24](conv3)   # Bx512x26x26
        conv5 = self.features[24:](conv4)     # Bx512x13x13
        
        # decode & upsample
        score = self.decoder(conv5)           # BxCx7x7
        upscore2 = self.upsample2(score)      # BxCx16x16
        
        score_pool4 = self.decoder_pool4(conv4)[:, :, 5:-5, 5:-5]  # BxCx16x16
        upscore_pool4 = self.upsample4(upscore2 + score_pool4)     # BxCx34x34
        
        score_pool3 = self.decoder_pool3(conv3)[:, :, 9:-9, 9:-9]  # BxCx34x34
        output = self.upsample8(upscore_pool4 + score_pool3)[:, :, 28:-28, 28:-28]  # BxCx224x224
        return output
    
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

def _vgg_fcn8s(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG_FCN8s(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        print('Loading pretrained model ...')
        state_dict = model_zoo.load_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16_fcn8s(pretrained=True, progress=True, **kwargs):
    return _vgg_fcn8s('vgg16', 'D', False, pretrained, progress, **kwargs)



"""
import pdb
batch_size = 4
x = torch.randn([batch_size, 3, 224, 224], dtype=torch.float).cuda()
model = vgg16_fcn8s(**{'num_classes': 7}).cuda()
output = model(x)
prob, pred = torch.max(torch.softmax(output, dim=1), dim=1)
pdb.set_trace()
"""
