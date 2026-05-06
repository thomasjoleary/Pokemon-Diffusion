from Schedule import calc_linear_betas_and_alphas
import torch
from UNet import UNet, TYPE_TO_IDX, RANK_TO_IDX, GENERATION_TO_IDX, TYPE_VOCAB, RANK_VOCAB, GENERATION_VOCAB
import matplotlib.pyplot as plt
import os

def reverse(image, step, pred_noise, betas, alphas):
    beta_t = betas[step].view(-1, 1, 1, 1)
    alpha_bar_t = alphas[step].view(-1, 1, 1, 1)
    alpha_bar_prev = alphas[step - 1].view(-1, 1, 1, 1) if step > 0 else torch.ones_like(alpha_bar_t)
    alpha_t = 1 - beta_t

    # compute the mean
    mean = (1 / torch.sqrt(alpha_t)) * (image - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)

    if step == 0:
        return mean
    
    # add noise scaled by posterior variance
    variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
    noise = torch.randn_like(image)
    return mean + torch.sqrt(variance) * noise

def sample_loop(model, conditions, betas, alphas, device, steps=1000, g_scale=7.5):
    img = torch.randn(1, 3, 256, 256).to(device)
    
    for step in reversed(range(steps)):
        with torch.no_grad():
            pred_noise = model.cfg_forward(img, torch.tensor([step]).to(device), conditions, g_scale)
            img = reverse(img, step, pred_noise, betas, alphas)
    
    return img

def checkpoint_load(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch

def generate(model, conditions, betas, alphas, device, chk_no, g_scale):
    image = sample_loop(model, conditions, betas, alphas, device, g_scale=g_scale)
    image = image.squeeze(0).permute(1, 2, 0)
    between_0_and_1 = (image.clamp(-1, 1) + 1) / 2
    plt.imshow(between_0_and_1.cpu().detach().numpy())
    os.makedirs(f'outputs/checkpoint_{chk_no}', exist_ok=True)
    t1 = TYPE_VOCAB[conditions['type1'].item()]
    t2 = TYPE_VOCAB[conditions['type2'].item()]
    rnk = RANK_VOCAB[conditions['rank'].item()]
    gen = GENERATION_VOCAB[conditions['generation'].item()]
    plt.savefig(f"outputs/checkpoint_{chk_no}/sample_{t1}_{t2}_{rnk}_{gen}_guidance_{g_scale}.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base_channels=64).to(device)
    chk_no = input("Checkpoint number: ")
    checkpoint_load(f'checkpoints/unet_epoch_{chk_no}.pt', model, device)
    model.eval()

    guidance_scale = float(input("Guidance scale (e.g. 3.0, 5.0, 7.5): "))

    s_or_b = input("Single or batch? (s/b): ")

    betas, alphas = calc_linear_betas_and_alphas()
    betas, alphas = betas.to(device), alphas.to(device)

    if (s_or_b.lower() == "s"):
        while True:
            # conditions
            type1 = input("Type 1: ")
            type2 = input("Type 2 (can be none): ")
            rank = input("Rank (ordinary, baby, legendary, mythical): ")
            rank = ["ordinary" if rank.lower() == "o" else "baby" if rank.lower() == "b" else "legendary" if rank.lower() == "l" else "mythical" if rank.lower() == "m" else rank.lower()]
            generation = input("Generation (i - ix): ")
            generation = "generation-" + generation if not generation.startswith("generation") else generation

            conditions = {
                "type1": torch.tensor([TYPE_TO_IDX[type1.lower()]]).to(device),
                "type2": torch.tensor([TYPE_TO_IDX[type2.lower()]]).to(device),
                "rank": torch.tensor([RANK_TO_IDX[rank.lower()]]).to(device),
                "generation": torch.tensor([GENERATION_TO_IDX[generation.lower()]]).to(device)
            }
            
            generate(model, conditions, betas, alphas, device, chk_no, guidance_scale)
            sample_again = input("Sample again? (y/n): ")
            if sample_again.lower() != "y":
                break
    else:
        types1 = [t.strip() for t in input("Type 1 list (comma separated): ").split(",")]
        types2 = [t.strip() for t in input("Type 2 list (comma separated, can be none): ").split(",")]
        ranks = [r.strip() for r in input("Rank list (comma separated, ordinary/baby/legendary/mythical): ").split(",")]
        ranks = ["ordinary" if r.lower() == "o" else "baby" if r.lower() == "b" else "legendary" if r.lower() == "l" else "mythical" if r.lower() == "m" else r.lower() for r in ranks]
        generations = [g.strip() for g in input("Generation list (comma separated, i - ix): ").split(",")]
        generations = ["generation-" + gen if not gen.startswith("generation") else gen for gen in generations]

        if (len(types1) != len(types2) or len(types1) != len(ranks) or len(types1) != len(generations)):
            print("All lists must have the same length!")
            exit(1)

        for cond in range(len(types1)):
            conditions = {
                "type1": torch.tensor([TYPE_TO_IDX[types1[cond].lower()]]).to(device),
                "type2": torch.tensor([TYPE_TO_IDX[types2[cond].lower()]]).to(device),
                "rank": torch.tensor([RANK_TO_IDX[ranks[cond].lower()]]).to(device),
                "generation": torch.tensor([GENERATION_TO_IDX[generations[cond].lower()]]).to(device)
            }
            
            generate(model, conditions, betas, alphas, device, chk_no, guidance_scale)

    