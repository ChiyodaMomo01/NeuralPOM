"""Direct next-state neural backbones, with no physical advance or correction."""
from __future__ import annotations
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from neuralpom.models.corrector_net import CorrectorNet


class TritonForecast(nn.Module):
    def __init__(self, channels, grid, settings):
        super().__init__()
        from neuralpom.models.ai_triton import Triton
        self.height, self.width = grid[:2]
        layers = settings["num_spatial_layers"]
        if layers < 2 or layers % 2:
            raise ValueError("Triton spatial layers must be positive and even")
        multiple = 2**(layers//2)
        self.padded = tuple(math.ceil(n/multiple)*multiple for n in grid[:2])
        self.network = Triton({}, shape_in=(channels, *self.padded), output_channels=channels,
            spatial_hidden_dim=settings["spatial_hidden_dim"], temporal_hidden_dim=settings["temporal_hidden_dim"],
            num_spatial_layers=layers, num_temporal_layers=settings["num_temporal_layers"],
            in_time_seq_length=1, out_time_seq_length=1)

    def forward(self, value):
        if value.shape[-2:] != (self.height, self.width):
            raise ValueError("AI model spatial shape differs")
        padded = F.pad(value, (0, self.padded[1]-self.width, 0, self.padded[0]-self.height), mode="replicate")
        return self.network(padded)[..., :self.height, :self.width]


def build_model(settings, codec):
    if settings["architecture"] == "triton":
        return TritonForecast(codec.channels, codec.grid, settings)
    if settings["architecture"] == "resunet":
        model = CorrectorNet(codec.channels, codec.channels, settings.get("hidden_dim", 256))
        model.checkpoint_blocks = settings.get("checkpoint_blocks", True)
        return model
    if settings["architecture"] in ("simvp", "convlstm", "pastnet"):
        return OfficialForecast(codec.channels, codec.grid, settings)
    raise ValueError("Unregistered AI backbone")


class SynchronizedEMAQuantizer(nn.Module):
    """Upstream EMA equations with stable buffers and a shared eight-rank codebook.

    Replacing Parameters inside forward (as upstream does) invalidates DDP's
    reducer/optimizer references. EMA statistics are averaged across ranks,
    retaining the per-example EMA scale. The forecast-only upstream objective
    and straight-through gradient are unchanged; no VQ loss is added.
    """
    def __init__(self, original):
        super().__init__()
        self._embedding_dim = original._embedding_dim
        self._num_embeddings = original._num_embeddings
        self._decay, self._epsilon = original._decay, original._epsilon
        self._commitment_cost = original._commitment_cost
        self.register_buffer("embedding", original._embedding.weight.detach().clone())
        self.register_buffer("ema_w", original._ema_w.detach().clone())
        self.register_buffer("ema_cluster_size", original._ema_cluster_size.detach().clone())

    def forward(self, inputs):
        value = inputs.permute(0, 2, 3, 1).contiguous()
        flat = value.reshape(-1, self._embedding_dim)
        distances = flat.square().sum(1, keepdim=True) + self.embedding.square().sum(1) - 2*flat@self.embedding.T
        indices = distances.argmin(1)
        encodings = F.one_hot(indices, self._num_embeddings).to(flat.dtype)
        quantized = (encodings@self.embedding).reshape_as(value)
        if self.training:
            with torch.no_grad():
                counts = encodings.sum(0)
                dw = encodings.T@flat.detach()
                if dist.is_initialized():
                    dist.all_reduce(counts); dist.all_reduce(dw)
                    counts.div_(dist.get_world_size()); dw.div_(dist.get_world_size())
                self.ema_cluster_size.mul_(self._decay).add_(counts, alpha=1-self._decay)
                n = self.ema_cluster_size.sum()
                self.ema_cluster_size.copy_((self.ema_cluster_size+self._epsilon)/(n+self._num_embeddings*self._epsilon)*n)
                self.ema_w.mul_(self._decay).add_(dw, alpha=1-self._decay)
                self.embedding.copy_(self.ema_w/self.ema_cluster_size[:, None])
        loss = self._commitment_cost*F.mse_loss(quantized.detach(), value)
        quantized = value+(quantized-value).detach()
        probabilities = encodings.mean(0)
        perplexity = torch.exp(-(probabilities*torch.log(probabilities+1e-10)).sum())
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class OfficialForecast(nn.Module):
    def __init__(self, channels, grid, settings):
        super().__init__()
        from .upstream import load_upstream
        source = load_upstream(settings)
        self.architecture = settings["architecture"]
        self.height, self.width = grid[:2]
        multiple = 16 if self.architecture == "pastnet" else 2**(settings["num_spatial_layers"]//2)
        if self.architecture == "convlstm": multiple = 1
        self.padded = tuple(math.ceil(n/multiple)*multiple for n in grid[:2])
        if self.architecture == "simvp":
            self.network = source.SimVP((1, channels, *self.padded),
                hid_S=settings["spatial_hidden_dim"], hid_T=settings["temporal_hidden_dim"],
                N_S=settings["num_spatial_layers"], N_T=settings["num_temporal_layers"], output_dim=channels)
        elif self.architecture == "convlstm":
            widths = settings["convlstm_hidden_dims"]
            self.network = source.ConvLSTM(channels, widths, (3, 3), len(widths), batch_first=True)
            self.readout = nn.Conv2d(widths[-1], channels, 1)
        else:
            self.network = source.PastNetModel((1, channels, *self.padded),
                hid_T=settings["temporal_hidden_dim"], N_T=settings["num_temporal_layers"],
                res_units=32, res_layers=2, embedding_nums=512, embedding_dim=64)
            dst = self.network.DST_module
            dst._vq_vae = SynchronizedEMAQuantizer(dst._vq_vae)

    def forward(self, value):
        if value.shape[-2:] != (self.height, self.width):
            raise ValueError("AI model spatial shape differs")
        padded = F.pad(value, (0, self.padded[1]-self.width, 0, self.padded[0]-self.height), mode="replicate")
        if self.architecture == "convlstm":
            # Each daily call resets hidden state, matching the one-state input contract.
            sequence, _ = self.network(padded[:, None])
            result = self.readout(sequence[:, 0])
        else:
            result = self.network(padded[:, None])[:, 0]
        return result[..., :self.height, :self.width]
