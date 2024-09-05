from torchsummary import summary
from models import *

if __name__ == '__main__':
    model = stackhourglass_bsconv_s(192).cuda()
    summary(model, [(3, 256, 512), (3, 256, 512)], batch_size=10)
