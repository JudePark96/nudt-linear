import torch


class ProjectionLinear(torch.nn.Linear):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, k=1, bias=True):
        super(ProjectionLinear, self).__init__(in_features, out_features, bias)
        self.k = k

    # Test set: Average loss: 1.1397, Accuracy: 9559/10000 (96%)
    def forward(self, input):
        assert input.shape[1] == self.weight.t().shape[0]
        w_unit = torch.norm(self.weight, p=1)

        cos_theta = (torch.mm(input, self.weight.t()) / (torch.norm(self.weight, p=2) * torch.norm(input, p=2)))
        sin_theta = torch.sqrt(1.0 - (cos_theta ** 2))

        out = 4 * (w_unit * cos_theta) * sin_theta

        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == '__main__':
    linear = ProjectionLinear(10, 32)
    sample_data = torch.randn((32, 10))
    linear(sample_data)
