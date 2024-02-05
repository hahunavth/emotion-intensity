import torch
import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, f1_score

from config import read_config
from dataset import get_es_loaders, get_loaders
from loss import EmotionIntensityLoss, mixup_criterion, rank_loss
from optimizer import create_optimizer, create_scheduler
from model import create_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", device)

configs = read_config()
train_loader, val_loader, mix_train_set, mix_val_set, _train_set, _val_set = get_es_loaders(configs, device)

model, _kwargs = create_model('rank')
model.to('cuda')

criterion = EmotionIntensityLoss()
criterion.enable_mixup_loss = True
criterion.enable_rank_loss = True

n_params = sum([p.numel() for p in model.parameters()])
print(f"Number of parameters: {n_params}")

optimizer, _kwargs = create_optimizer('adam', model.parameters())
lr_scheduler, _kwargs = create_scheduler('linear', optimizer)

# Training loop
n_epochs = 200
step = 0
_bar = tqdm.tqdm(range(n_epochs))
for epoch in _bar:
    model.train()
    
    # exp3
    train_loader.dataset.alpha = 1.
    # exp2
    # train_loader.dataset.alpha = epoch / 50 if epoch < 50 else 1.
    # exp1
    # if epoch <= 1:
    #     train_loader.dataset.alpha = 0.
    #     train_loader.dataset.rand_lam_per_batch = True
    # if epoch == 10:
    #     train_loader.dataset.alpha = 0.1
    # if epoch == 20:
    #     train_loader.dataset.alpha = 0.2
    # if epoch == 25:
    #     train_loader.dataset.alpha = 0.5
    #     train_loader.dataset.rand_lam_per_batch = False
    # if epoch == 30:
        # train_loader.dataset.alpha = 1.
    print("Alpha: ", train_loader.dataset.alpha)
    
    total_losses = None
    for idx, batch in enumerate(train_loader):
        model.train()
        xi, xj, lam_i, lam_j, xi_lens, xj_lens, y_neu, y_emo = batch
        
        xi = xi.to(device)
        xj = xj.to(device)
        lam_i = lam_i.to(device)
        lam_j = lam_j.to(device)
        y_neu = y_neu.to(device)
        y_emo = y_emo.to(device)
        
        ii, hi, ri, _hi = model(xi, y_emo)
        ij, hj, rj, _hi = model(xj, y_emo)
        
        loss, losses = criterion((ii, hi, ri), (ij, hj, rj), y_emo, y_neu, lam_i, lam_j)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # losses = {
        #     "total": loss.item(),
        #     "mi": mi.item(), 
        #     "mj": mj.item(), 
        #     "rank": ranking_loss.item(),
        # }
        if total_losses is None:
            total_losses = losses
        else:
            for k, v in losses.items():
                total_losses[k] += v
        
        # step log
        if idx % 100 == 0 and idx > 0:
            print("Step", step, "losses: ", {k: v / idx for k, v in total_losses.items()})
    
        # end step
        step += 1
        
    # scheduler.step()
    msg = "Epoch: {}/{}, Losses: {}, Lr: {}"\
        .format(
            (epoch+1), n_epochs, 
            {k: v / len(train_loader) for k, v in total_losses.items()}, 
            optimizer.param_groups[0]['lr']
        )
    # msg = f"Epoch: ", (epoch+1)/epochs, "," \
    #         "Losses:", {k: v / len(train_loader) for k, v in total_losses.items()}, ","\
    #         "LR:", optimizer.param_groups[0]['lr'], ","
    print(msg)
    
    # Validation loop
    # model.eval()
    # total_val_loss = 0
    # total_rl_val_loss = 0
    # total_mx_val_loss = 0
    # with torch.no_grad():
    #     for val_batch in val_loader:
    #         xi_val, xj_val, lam_i_val, lam_j_val, xi_lens, xj_lens, y_neu_val, y_emo_val = val_batch
            
    #         xi_val = xi_val.to(device)
    #         xj_val = xj_val.to(device)
    #         lam_i_val = lam_i_val.to(device)
    #         lam_j_val = lam_j_val.to(device)
    #         y_neu_val = y_neu_val.to(device)
    #         y_emo_val = y_emo_val.to(device)
            
    #         prediction_val = model(xi_val, xj_val)
    #         loss_val, loss_val_mx, loss_val_rl = model.compute_loss(prediction_val, y_emo_val, y_neu_val, lam_i_val, lam_j_val)
    #         total_val_loss += loss_val.item()
    #         total_rl_val_loss += loss_val_rl.item()
    #         total_mx_val_loss += loss_val_mx.item()
    # msg2 = f"Validation Loss: {total_val_loss / len(val_loader)}, Mixup Loss: {total_mx_val_loss / len(val_loader)}, Rank Loss: {total_rl_val_loss / len(val_loader)}"
    # print(msg)
    # print()
    # validation
    msg2 = ""
    if epoch % 1 == 0:
        model.eval()
        emo_lbs = []
        emo_preds = []
        for idx in range(len(mix_train_set.wrapped_ds)):
            sample = mix_train_set.get_one_fn(idx)
            if "speaker" in sample and sample['speaker'] != 0:
                continue
            mels = torch.from_numpy(sample["mel"]).unsqueeze(0)
            emo = sample["emotion"]
            mels = mels.to(device)
            
            i, h, r, _ = model(mels)
            
            emo_lbs.append(emo)
            emo_preds.append(torch.argmax(h, dim=1).item())
            # print(h)
            # exit()
        acc = accuracy_score(emo_lbs, emo_preds)
        f1 = f1_score(emo_lbs, emo_preds, average=None)
        msg2 = f'Train Accuracy: {acc}, F1: {f1}'
        print(msg2)
    
    msg2 = ""
    if epoch % 1 == 0:
        model.eval()
        emo_lbs = []
        emo_preds = []
        for idx in range(len(mix_val_set.wrapped_ds)):
            sample = mix_val_set.get_one_fn(idx)
            if "speaker" in sample and sample['speaker'] != 0:
                continue
            mels = torch.from_numpy(sample["mel"]).unsqueeze(0)
            emo = sample["emotion"]
            mels = mels.to(device)
            
            i, h, r, _ = model(mels)
            
            emo_lbs.append(emo)
            emo_preds.append(torch.argmax(h, dim=1).item())
            # print(h)
            # exit()
        acc = accuracy_score(emo_lbs, emo_preds)
        f1 = f1_score(emo_lbs, emo_preds, average=None)
        msg2 = f'Accuracy: {acc}, F1: {f1}'
        print(msg2)
    
    with open("log.txt", "a") as f:
        f.write(msg)
        f.write("\n")
        f.write(msg2)
        f.write("\n")
    
    if epoch % 5 == 0:
        torch.save({"state_dict": model.state_dict()}, f"rank_model_{epoch}.pt")
        
