import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import kornia

from ...modules.x_transformer import Encoder, TransformerWrapper


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast
        # self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizerFast.from_pretrained('./models/bert')
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 strides=[],
                 method='bilinear',
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.strides = strides
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method, align_corners=True)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for h_s, w_s in self.strides:
            x = self.interpolator(x, scale_factor=(1/h_s, 1/w_s))

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.model.to(device)
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipMultiTextEmbedder(FrozenCLIPTextEmbedder):
    def __init__(self, num_views=1, apply_all=False, **kwargs):
        super().__init__(**kwargs)
        self.num_views = num_views
        self.apply_all = apply_all

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]

        if not self.apply_all:
            new_z = torch.zeros(*z.shape[:2], z.shape[2] * self.num_views, device=z.device)
            new_z[:, :, self.num_views // 2 * z.shape[2]: (self.num_views // 2 + 1) * z.shape[2]] = z
        else:
            new_z = repeat(z, 'b 1 d -> b 1 (d m)', m=self.num_views)

        return new_z


class FrozenClipImageEmbedder(nn.Module):
    """
    Uses the CLIP image encoder.
    """

    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device='cpu', jit=jit)
        self.init()

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def init(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        # x = (x + 1.) / 2.

        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [0,1]
        return self.model.encode_image(self.preprocess(x))


class FrozenClipMultiImageEmbedder(FrozenClipImageEmbedder):
    """
    Uses the CLIP image encoder with multi-image as input.
    """

    def __init__(self, num_views=1, split_per_view=1, img_dim=768, out_dim=512, key='camera', **kwargs):
        super().__init__(**kwargs)
        self.split_per_view = split_per_view
        self.key = key
        self.linear = nn.Linear(img_dim, out_dim)
        self.view_embedding = nn.Parameter(img_dim ** -0.5 * torch.randn((1, num_views * split_per_view, img_dim)))

    def forward(self, x):
        # x is assumed to be in range [0,1]
        if isinstance(x, torch.Tensor) and x.ndim == 5:
            x = x.permute(1, 0, 2, 3, 4)
        elif isinstance(x, dict):
            x = x[self.key]
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            x = self.linear(x)
            return x

        with torch.no_grad():
            img_feats = [self.model.encode_image(self.preprocess(img))[:, None] for img in x]
            x = torch.cat(img_feats, 1).float() + self.view_embedding
            x = self.linear(x)

        return x


class FrozenClipImagePatchEmbedder(nn.Module):
    """
    Uses the CLIP image encoder.
    """

    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
            img_dim=1024,
            out_dim=512,
            num_views=1,
            split_per_view=1
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device='cpu', jit=jit)
        self.init()

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.view_embedding = nn.Parameter(img_dim ** -0.5 * torch.randn((1, num_views * split_per_view, 1, img_dim)))

        self.linear = nn.Linear(img_dim, out_dim)

    def init(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        # x = (x + 1.) / 2.

        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def encode_image_patch(self, x):
        visual_encoder = self.model.visual
        x = x.type(self.model.dtype)
        x = visual_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual_encoder.positional_embedding.to(x.dtype)
        x = visual_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:, :]

        return x

    def forward(self, x):
        # x is assumed to be in range [0,1]
        img_feats = [self.encode_image_patch(self.preprocess(img))[:, None] for img in x]
        x = torch.cat(img_feats, 1).float() + self.view_embedding
        x = rearrange(x, 'b v n c -> b (v n) c')
        x = self.linear(x)
        return x
