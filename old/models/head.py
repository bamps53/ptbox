import torch
import torch.nn as nn

class CustomHead(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.25),
            nn.Linear(in_features, in_features // 2),
            nn.PReLU(),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(0.125),
            nn.Linear(in_features // 2, out_features),
        )

class MultiSampleDropout(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.num_samples = num_samples
        for i in range(num_samples):
            setattr(self, 'dropout{}'.format(i), nn.Dropout(dropout_rate))
            setattr(self, 'fc{}'.format(i), nn.Linear(in_features, out_features))

    def forward(self, x):
        logits = []
        for i in range(self.num_samples):
            dropout = getattr(self, 'dropout{}'.format(i))
            fc = getattr(self, 'fc{}'.format(i))
            x_ = dropout(x)
            x_ = fc(x_)
            logits.append(x_)
        return torch.stack(logits).mean(dim=0)

class MultiHead(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        self.num_samples = num_samples
        # self.g_head = CustomHead(in_features, self.n_grapheme)
        # self.v_head = CustomHead(in_features, self.n_vowel)
        # self.c_head = CustomHead(in_features, self.n_consonant)
        self.g_head = nn.Sequential(
            # nn.BatchNorm1d(in_features),
            MultiSampleDropout(in_features, in_features // 2, num_samples=5, dropout_rate=0.5),
            nn.ReLU(),
            # nn.BatchNorm1d(in_features // 2),
            MultiSampleDropout(in_features // 2, self.n_grapheme, num_samples=5, dropout_rate=0.5),
        )
        self.v_head = nn.Sequential(
            # nn.BatchNorm1d(in_features),
            MultiSampleDropout(in_features, in_features // 2, num_samples=5, dropout_rate=0.5),
            nn.ReLU(),
            # nn.BatchNorm1d(in_features // 2),
            MultiSampleDropout(in_features // 2, self.n_vowel, num_samples=5, dropout_rate=0.5),
        )
        self.c_head = nn.Sequential(
            # nn.BatchNorm1d(in_features),
            MultiSampleDropout(in_features, in_features // 2, num_samples=5, dropout_rate=0.5),
            nn.ReLU(),
            # nn.BatchNorm1d(in_features // 2),
            MultiSampleDropout(in_features // 2, self.n_consonant, num_samples=5, dropout_rate=0.5),
        )

    def forward(self, x):
        g_logits = self.g_head(x)
        v_logits = self.v_head(x)
        c_logits = self.c_head(x)
        return torch.cat([g_logits, v_logits, c_logits], dim=1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def msd_layer(self, in_features, out_features, num_samples, dropout_rate):
    return nn.Sequential(
        MultiSampleDropout(in_features, in_features // 2, num_samples=num_samples, dropout_rate=dropout_rate),
        nn.ReLU(),
        MultiSampleDropout(in_features // 2, out_features, num_samples=num_samples, dropout_rate=dropout_rate),
    )

class AuxHead(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        self.num_samples = num_samples
        self.g_head1 = msd_layer(in_features, self.n_grapheme, num_samples=num_samples, dropout_rate=dropout_rate)
        self.g_head2 = msd_layer(in_features, self.n_grapheme, num_samples=num_samples, dropout_rate=dropout_rate)
        self.v_head1 = msd_layer(in_features, self.n_vowel, num_samples=num_samples, dropout_rate=dropout_rate)
        self.v_head2 = msd_layer(in_features, self.n_vowel, num_samples=num_samples, dropout_rate=dropout_rate)
        self.c_head1 = msd_layer(in_features, self.n_consonant, num_samples=num_samples, dropout_rate=dropout_rate)
        self.c_head2 = msd_layer(in_features, self.n_consonant, num_samples=num_samples, dropout_rate=dropout_rate)

    def forward(self, x):
        g_logits1 = self.g_head1(x)
        v_logits1 = self.v_head1(x)
        c_logits1 = self.c_head1(x)
        x1 = torch.cat([g_logits1, v_logits1, c_logits1], dim=1)
        g_logits2 = self.g_head(x1)
        v_logits2 = self.v_head(x1)
        c_logits2 = self.c_head(x1)
        x2 = torch.cat([g_logits2, v_logits2, c_logits2], dim=1)
        return  torch.cat([x1, x2], dim=1)


