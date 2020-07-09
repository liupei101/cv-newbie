"""
DenseNet in Pytorch

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn

class Bottleneck(nn.Module):
	def __init__(self, in_channels, growth_rate):
		super().__init__()

		# In  our experiments, we let each 1×1 convolution 
        # produce 4k feature-maps
		btneck_channels = 4 * growth_rate

        # Note that each “conv” layer shown in the
        # table corresponds the sequence BN-ReLU-Conv
		self.bottle_neck = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, btneck_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(btneck_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(btneck_channels, growth_rate, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		# concate along channel-axis (H W keep unchanged)
		return torch.cat([x, self.bottle_neck(x)], 1)

# Transition_A: Splition when inputs coming
# each input splition go through BN-Conv-Pool
class Transition_A(nn.Module):
	def __init__(self, in_channels_branch, out_channels_branch):
		super().__init__()

		for i in range(len(in_channels_branch) - 1):
			assert in_channels_branch[i + 1] == in_channels_branch[1]

		self.in_channels_branch = in_channels_branch
		self.out_channels_branch = out_channels_branch

		self.down_sample_first = nn.Sequential(
			nn.BatchNorm2d(in_channels_branch[0]),
			nn.Conv2d(in_channels_branch[0], out_channels_branch[0], 1, bia=False),
			nn.AvgPool2d(2, stride=2)
		)

		self.down_sample_subsq = nn.Sequential(
			nn.BatchNorm2d(in_channels_branch[1]),
			nn.Conv2d(in_channels_branch[1], out_channels_branch[1], 1, bia=False),
			nn.AvgPool2d(2, stride=2)
		)

	def forward(self, x):
		x = x.split(self.in_channels_branch, dim=1)
		
		return torch.cat(
			[self.down_sample_first(x[0])] + [self.down_sample_subsq(x[i+1]) for i in range(len(in_channels_branch) - 1)], 
			dim=1
		)

# Transition_B: Splition when computing 1x1 Conv and after BatchNorm layer
# BN - each splition go through Conv - Pool
class Transition_B(nn.Module):
	def __init__(self, in_channels_branch, out_channels_branch):
		super().__init__()

		for i in range(len(in_channels_branch) - 1):
			assert in_channels_branch[i + 1] == in_channels_branch[1]

		self.in_channels_branch = in_channels_branch
		self.out_channels_branch = out_channels_branch

		self.bn = nn.BatchNorm2d(sum(in_channels_branch))
		self.conv_first = nn.Conv2d(in_channels_branch[0], out_channels_branch[0], 1, bia=False),
		self.conv_subsq = nn.Conv2d(in_channels_branch[1], out_channels_branch[1], 1, bia=False),
		self.avgpool = nn.AvgPool2d(2, stride=2)

	def forward(self, x):
		output = self.bn(x)
		
		# weighted 1x1 Conv
		output = output.split(self.in_channels_branch, dim=1)
		output = torch.cat(
			[self.conv_first(output[0])] + [self.conv_subsq(output[i+1]) for i in range(len(in_channels_branch) - 1)], 
			dim=1
		)
		
		output = self.avgpool(output)

		return output

# DesneNet-BC
# B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
# C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
	def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
		super.__init__()
		self.growth_rate = growth_rate

		inner_channels = 2 * growth_rate

		self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

		self.features = nn.Sequential()

		for index in range(len(nblocks) - 1):
			self.features.add_module(
				"dense_block_layer_{}".format(index + 1),
				self._make_dense_layers(block, inner_channels, nblocks[index])
			)
			# k_0 + k * L
			inner_channels = inner_channels + growth_rate * nblocks[index]

			# compression
			out_channels = int(reduction * inner_channels)
			# transition
			self.features.add_module(
				"transition_layer_{}".format(index), 
				Transition_B(inner_channels, out_channels)
			)
			inner_channels = out_channels

        # The last Dense Block
		self.features.add_module(
			"dense_block_layer_{}".format(len(nblocks)), 
			self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1])
		)
		inner_channels += growth_rate * nblocks[len(nblocks) - 1]

		self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
		self.features.add_module('relu', nn.ReLU(inplace=True))

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.linear = nn.Linear(inner_channels, num_class)

	def forward(self, x):
		output = self.conv1(x)
		output = self.features(output)
		output = self.avgpool(output)
		output = output.view(output.size()[0], -1)
		output = self.linear(output)
		return output

	def _make_dense_layers(self, block, in_channels, nblocks):
		dense_block = nn.Sequential()
		for index in range(nblocks):
			dense_block.add_module(
				'bottle_neck_layer_{}'.format(index + 1), 
				block(in_channels, self.growth_rate)
				)
			in_channels += self.growth_rate

		return dense_block


def densenet121():
	return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
	return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
	return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
	return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)