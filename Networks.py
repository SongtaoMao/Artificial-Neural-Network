from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=11, output_dim=1):
        super(MLP, self).__init__()

        self.output = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim),
        )

    def forward(self, x):
        return self.output(x)