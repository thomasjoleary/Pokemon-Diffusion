import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# vocab lists must match CSV values, "none" at index 0 covers pokemon with no second type
TYPE_VOCAB = ["none", "bug", "dark", "dragon", "electric", "fairy", "fighting",
              "fire", "flying", "ghost", "grass", "ground", "ice",
              "normal", "poison", "psychic", "rock", "steel", "water"]
RANK_VOCAB = ["ordinary", "baby", "legendary", "mythical"]
GENERATION_VOCAB = ["generation-i", "generation-ii", "generation-iii", "generation-iv",
                    "generation-v", "generation-vi", "generation-vii", "generation-viii", "generation-ix"]

# reverse lookup from string to integer index
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPE_VOCAB)}
RANK_TO_IDX = {r: i for i, r in enumerate(RANK_VOCAB)}
GENERATION_TO_IDX = {g: i for i, g in enumerate(GENERATION_VOCAB)}


def build_condition(type1, type2, rank, generation, device=torch.device("cpu")):
    # converts raw CSV strings into integer tensors the model expects
    t2 = type2.lower() if type2 and type2.lower() != "none" else "none"
    return {
        "type1": torch.tensor([TYPE_TO_IDX[type1.lower()]], device=device),
        "type2": torch.tensor([TYPE_TO_IDX[t2]], device=device),
        "rank": torch.tensor([RANK_TO_IDX[rank.lower()]], device=device),
        "generation": torch.tensor([GENERATION_TO_IDX[generation.lower()]], device=device),
    }


class SinusoidalPosEmb(nn.Module):
    # encodes the timestep as a fixed vector so the model knows how noisy the input is
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConditionEmbedder(nn.Module):
    # embeds type1, type2, rank and generation then concatenates them into one vector
    # each table has an extra null token at the end used when a condition is dropped for CFG
    def __init__(self, embed_dim=128, cond_drop_prob=0.10):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.type_emb = nn.Embedding(len(TYPE_VOCAB) + 1, embed_dim)  # type1 and type2 share this table
        self.rank_emb = nn.Embedding(len(RANK_VOCAB) + 1, embed_dim)
        self.gen_emb = nn.Embedding(len(GENERATION_VOCAB) + 1, embed_dim)
        self.register_buffer("null_type", torch.tensor([len(TYPE_VOCAB)]))
        self.register_buffer("null_rank", torch.tensor([len(RANK_VOCAB)]))
        self.register_buffer("null_gen", torch.tensor([len(GENERATION_VOCAB)]))
        self.out_dim = embed_dim * 4

    def maybe_drop(self, idx, null_idx):
        # randomly replace a condition with its null token during training so the model learns to run without it
        if self.training and self.cond_drop_prob > 0:
            mask = torch.rand(idx.shape, device=idx.device) < self.cond_drop_prob
            idx = torch.where(mask, null_idx.expand_as(idx), idx)
        return idx

    def forward(self, cond, force_uncond=False):
        t1, t2, rnk, gen = cond["type1"], cond["type2"], cond["rank"], cond["generation"]
        if force_uncond:
            # null out everything for the unconditional pass during CFG inference
            B = t1.shape[0]
            t1 = self.null_type.expand(B)
            t2 = self.null_type.expand(B)
            rnk = self.null_rank.expand(B)
            gen = self.null_gen.expand(B)
        else:
            t1 = self.maybe_drop(t1, self.null_type)
            t2 = self.maybe_drop(t2, self.null_type)
            rnk = self.maybe_drop(rnk, self.null_rank)
            gen = self.maybe_drop(gen, self.null_gen)
        return torch.cat([self.type_emb(t1), self.type_emb(t2),
                          self.rank_emb(rnk), self.gen_emb(gen)], dim=-1)


class ResBlock(nn.Module):
    # two conv layers with a skip connection, conditioned on the time and condition embedding
    def __init__(self, in_ch, out_ch, emb_dim, groups=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))  # predicts scale and shift
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)  # inject time and condition info
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.dropout(self.conv2(F.silu(self.norm2(h))))
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    # lets every spatial position attend to every other position, used at lower resolutions only
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)
        q, k, v = q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


# builds the two resblocks and optional attention layer used by both encoder and decoder stages
def make_res_stack(in_ch, out_ch, emb_dim, use_attn, groups):
    return nn.ModuleList([
        ResBlock(in_ch, out_ch, emb_dim, groups),
        ResBlock(out_ch, out_ch, emb_dim, groups),
        SelfAttention(out_ch) if use_attn else nn.Identity(),
    ])


def run_res_stack(stack, x, emb):
    # passes x through res1, res2, then attn (attn does not take emb)
    x = stack[0](x, emb)
    x = stack[1](x, emb)
    x = stack[2](x)
    return x


class DownBlock(nn.Module):
    # one encoder stage, halves resolution and saves a skip connection for the decoder
    def __init__(self, in_ch, out_ch, emb_dim, use_attn=False, groups=8):
        super().__init__()
        self.stack = make_res_stack(in_ch, out_ch, emb_dim, use_attn, groups)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, emb):
        x = run_res_stack(self.stack, x, emb)
        return self.down(x), x  # downsampled output and skip connection


class UpBlock(nn.Module):
    # one decoder stage, doubles resolution and merges the skip from the matching encoder stage
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim, use_attn=False, groups=8):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                nn.Conv2d(in_ch, in_ch, 3, padding=1))
        self.stack = make_res_stack(in_ch + skip_ch, out_ch, emb_dim, use_attn, groups)

    def forward(self, x, skip, emb):
        x = torch.cat([self.up(x), skip], dim=1)
        return run_res_stack(self.stack, x, emb)


class UNet(nn.Module):
    # predicts the noise in a noisy image given a timestep and pokemon conditions
    # encoder compresses down to 16x16, bottleneck processes it, decoder expands back to 256x256
    # skip connections carry spatial detail from encoder to decoder at each resolution
    def __init__(self, image_channels=3, base_channels=128, channel_mults=(1, 2, 4, 8),
                 time_emb_dim=256, cond_embed_dim=128, cond_drop_prob=0.10, groups=8):
        super().__init__()
        ch = [base_channels * m for m in channel_mults]  # e.g. [128, 256, 512, 1024]

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.cond_emb = ConditionEmbedder(cond_embed_dim, cond_drop_prob)
        # projects condition vector into the same size as the time embedding so they can be summed
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_emb.out_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        emb_dim = time_emb_dim

        self.stem = nn.Conv2d(image_channels, ch[0], 3, padding=1)  # lifts RGB to feature channels

        # attention is disabled at the two highest resolutions to keep memory manageable
        self.down0 = DownBlock(ch[0], ch[0], emb_dim, use_attn=False, groups=groups)  # 256->128
        self.down1 = DownBlock(ch[0], ch[1], emb_dim, use_attn=False, groups=groups)  # 128->64
        self.down2 = DownBlock(ch[1], ch[2], emb_dim, use_attn=False, groups=groups)  # 64->32
        self.down3 = DownBlock(ch[2], ch[3], emb_dim, use_attn=True,  groups=groups)  # 32->16

        # bottleneck at 16x16
        self.mid_res1 = ResBlock(ch[3], ch[3], emb_dim, groups)
        self.mid_attn = SelfAttention(ch[3])
        self.mid_res2 = ResBlock(ch[3], ch[3], emb_dim, groups)

        self.up3 = UpBlock(ch[3], ch[3], ch[2], emb_dim, use_attn=True,  groups=groups)  # 16->32
        self.up2 = UpBlock(ch[2], ch[2], ch[1], emb_dim, use_attn=False, groups=groups)  # 32->64
        self.up1 = UpBlock(ch[1], ch[1], ch[0], emb_dim, use_attn=False, groups=groups)  # 64->128
        self.up0 = UpBlock(ch[0], ch[0], ch[0], emb_dim, use_attn=False, groups=groups)  # 128->256

        self.out = nn.Sequential(nn.GroupNorm(groups, ch[0]), nn.SiLU(),
                                 nn.Conv2d(ch[0], image_channels, 3, padding=1))

    def embed(self, t, cond, force_uncond=False):
        # build a single embedding from timestep and conditions, passed to every ResBlock
        return self.time_emb(t) + self.cond_proj(self.cond_emb(cond, force_uncond))

    def unet_body(self, x, emb):
        # full encoder, bottleneck, decoder pass given a pre-computed embedding
        h = self.stem(x)
        h, s0 = self.down0(h, emb)
        h, s1 = self.down1(h, emb)
        h, s2 = self.down2(h, emb)
        h, s3 = self.down3(h, emb)
        h = self.mid_res2(self.mid_attn(self.mid_res1(h, emb)), emb)
        h = self.up3(h, s3, emb)
        h = self.up2(h, s2, emb)
        h = self.up1(h, s1, emb)
        h = self.up0(h, s0, emb)
        return self.out(h)

    def forward(self, x, t, cond):
        return self.unet_body(x, self.embed(t, cond))

    @torch.no_grad()
    def cfg_forward(self, x, t, cond, guidance_scale=5.0):
        # runs the model with and without conditions, then blends the two outputs
        # higher guidance_scale pushes the result to more strongly match the conditions
        emb_cond = self.embed(t, cond, force_uncond=False)
        emb_uncond = self.embed(t, cond, force_uncond=True)
        emb2 = torch.cat([emb_cond, emb_uncond], dim=0)
        x2 = torch.cat([x, x], dim=0)
        out = self.unet_body(x2, emb2)
        eps_cond, eps_uncond = out.chunk(2, dim=0)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build a smaller version of the model for testing and print how many learnable parameters it has
    model = UNet(base_channels=64, cond_embed_dim=64).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # test fake inputs - two random noisy images, a random timestep for each, and hand-picked conditions
    B = 2
    x = torch.randn(B, 3, 256, 256).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    cond = {
        "type1": torch.tensor([TYPE_TO_IDX["fire"], TYPE_TO_IDX["water"]]).to(device),
        "type2": torch.tensor([TYPE_TO_IDX["none"], TYPE_TO_IDX["flying"]]).to(device),
        "rank": torch.tensor([RANK_TO_IDX["ordinary"], RANK_TO_IDX["legendary"]]).to(device),
        "generation": torch.tensor([GENERATION_TO_IDX["generation-i"],
                                    GENERATION_TO_IDX["generation-iii"]]).to(device),
    }

    # training mode - runs a single conditioned forward pass and checks the output shape matches the input
    model.train()
    print(f"Train output: {model(x, t, cond).shape}")

    # eval mode - runs the CFG forward pass which blends conditioned and unconditioned outputs
    # checks the output shape is still correct after the double batch trick
    model.eval()
    print(f"CFG output: {model.cfg_forward(x, t, cond, guidance_scale=7.5).shape}")

    # to use the UNet in another file:
    #
    #   from UNet import UNet, build_condition
    #
    #   model = UNet(base_channels=128, cond_embed_dim=128).to(device)
    #   checkpoint = torch.load("checkpoints/unet.pt", map_location=device)
    #   model.load_state_dict(checkpoint["model"])
    #   model.eval()
    #
    #   cond = build_condition(type1="fire", type2="none", rank="legendary", generation="generation-i", device=device)
    #
    #   x = torch.randn(1, 3, 256, 256).to(device)
    #   t = torch.tensor([999]).to(device)
    #   output = model.cfg_forward(x, t, cond, guidance_scale=7.5)
