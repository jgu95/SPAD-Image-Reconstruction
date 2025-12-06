import torch
import torch.nn as nn

# ================= Basic Modules =================
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(nn.Linear(c, c//r), nn.ReLU(True), nn.Linear(c//r, c), nn.Sigmoid())
    def forward(self, x): b,c,_,_=x.size(); return x*self.ex(self.sq(x).view(b,c)).view(b,c,1,1)

class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch,out_ch,3,1,1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
                                  nn.Conv2d(out_ch,out_ch,3,1,1), nn.BatchNorm2d(out_ch))
        self.sc=nn.Conv2d(in_ch,out_ch,1); self.se=SEBlock(out_ch); self.act=nn.ReLU(True)
    def forward(self, x): return self.act(self.se(self.conv(x))+self.sc(x))

# ================= Stage 1 Model (Reconstruction) =================
class ResUNet(nn.Module):
    def __init__(self, in_ch=48, out_ch=3, base_ch=64):
        super().__init__()
        self.inc = ResConv(in_ch, base_ch)
        self.d1=nn.Sequential(nn.MaxPool2d(2),ResConv(base_ch,base_ch*2))
        self.d2=nn.Sequential(nn.MaxPool2d(2),ResConv(base_ch*2,base_ch*4))
        self.d3=nn.Sequential(nn.MaxPool2d(2),ResConv(base_ch*4,base_ch*8))
        self.d4=nn.Sequential(nn.MaxPool2d(2),ResConv(base_ch*8,base_ch*16))
        self.u1=nn.ConvTranspose2d(base_ch*16,base_ch*8,2,2); self.c1=ResConv(base_ch*16,base_ch*8)
        self.u2=nn.ConvTranspose2d(base_ch*8,base_ch*4,2,2); self.c2=ResConv(base_ch*8,base_ch*4)
        self.u3=nn.ConvTranspose2d(base_ch*4,base_ch*2,2,2); self.c3=ResConv(base_ch*4,base_ch*2)
        self.u4=nn.ConvTranspose2d(base_ch*2,base_ch,2,2); self.c4=ResConv(base_ch*2,base_ch)
        self.out=nn.Conv2d(base_ch,out_ch,1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x=self.c1(torch.cat([self.u1(x5),x4],1)); x=self.c2(torch.cat([self.u2(x),x3],1))
        x=self.c3(torch.cat([self.u3(x),x2],1)); x=self.c4(torch.cat([self.u4(x),x1],1))
        return torch.sigmoid(self.out(x))

# ================= Stage 2 Model (Refinement) =================
# Structure is the same as ResUNet; this class is defined to avoid errors when loading weights
class RefineUNet(ResUNet):
    def __init__(self):
        super().__init__(in_ch=3, out_ch=3, base_ch=64)
    
    def forward(self, x):
        # Note: RefineUNet implements residual connection internally during training
        # However, in inference scripts, we usually add manually or call this forward method
        # For compatibility, here we return the result after adding the residual
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x_dec=self.c1(torch.cat([self.u1(x5),x4],1)); x_dec=self.c2(torch.cat([self.u2(x_dec),x3],1))
        x_dec=self.c3(torch.cat([self.u3(x_dec),x2],1)); x_dec=self.c4(torch.cat([self.u4(x_dec),x1],1))
        return x + self.out(x_dec) # Residual Learning