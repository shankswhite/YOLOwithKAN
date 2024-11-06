
class C2f_KAN_1D(nn.Module):
    """Faster Implementation of CSP Bottleneck with KAN-enhanced convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.flatten = FlattenLayer(patch_size=7, stride=1, in_chans=self.c, embed_dim=self.c)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_KAN_1D(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f_KAN_1D layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split into two parts

        # Apply flattening to y[-1]
        flattened_y, H, W = self.flatten(y[-1])  # Shape: [B, N, C]

        # Pass flattened_y through each Bottleneck_KAN_1D layer
        for m in self.m:
            flattened_y = m(flattened_y, H, W)  # The output will be [B, c2, H, W]
            # Flatten the output again for the next iteration
            flattened_y = flattened_y.flatten(2).transpose(1, 2)  # Shape: [B, N, c2]

        # After the loop, reshape the output to [B, c2, H, W] for concatenation
        y.append(flattened_y.transpose(1, 2).view(-1, self.c, H, W))

        # Concatenate along the channel dimension
        return self.cv2(torch.cat(y, 1))


class Bottleneck_KAN_1D(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = KAN_Block(c1, c_, k[0])
        self.cv2 = KAN_Block(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x, H, W):
        out = self.cv1(x, H, W)
        out = self.cv2(out, H, W)
        return x + out if self.add else out


class FlattenLayer(nn.Module):
    """Image to Patch Embedding with 3x3 convolution to flatten input for KAN_Block."""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights (optional, based on PatchEmbed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.proj(x)  # Shape: [B, embed_dim, H, W]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, N, embed_dim], N = H * W
        x = self.norm(x)
        return x, H, W

class KAN_Block(nn.Module):
    """KAN Block with KANLayer, DWConv, BatchNorm, and SiLU activation."""

    def __init__(self, c1, c2, k=3):
        super(KAN_Block, self).__init__()
        # KANLayer: c1 to c2 transformation
        self.kanlayer = KAN([c1, c2], grid_size=4, spline_order=3, scale_noise=0.1)
        # Depthwise Convolution
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=k, stride=1, padding=k // 2, groups=c2)
        # Batch Normalization and Activation
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x, H, W):
        # x is of shape [B, N, C], where N = H * W
        B, N, C = x.shape

        # Apply KANLayer
        x = x.view(-1, C)  # Shape: [B * N, C]
        x = self.kanlayer(x)  # Shape: [B * N, c2]

        # Reshape back to [B, c2, H, W]
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # Shape: [B, c2, H, W]

        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
