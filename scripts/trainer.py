import os
import torch
import scipy
from tqdm import tqdm
from torch.utils.data import DataLoader
from scripts.DataLoad_Train import generate_random_bool_masked_pos


def train_and_validate(
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler,
        config,
        args,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        epochs=20,
        mask_ratio=0.7,
        save_last=True,
        ckpt_dir='./models',
        loss_dir='./results/Loss'
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    model.to(device)
    loss_all_train = []
    loss_all_val = []

    best_val = float('inf')
    best_epoch = -1
    best_ckpt_path = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pixel_values = batch.to(device)

            bool_masked_pos = generate_random_bool_masked_pos(
                batch_size=pixel_values.size(0),
                num_patches=config.time_num//config.patch_size,
                mask_ratio=mask_ratio,
            ).to(device)

            outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新参数和学习率
            if scheduler is not None:
                scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        loss_all_train.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch.to(device)
                bool_masked_pos = generate_random_bool_masked_pos(
                    batch_size=pixel_values.size(0),
                    num_patches=config.time_num//config.patch_size,
                    mask_ratio=mask_ratio,
                ).to(device)

                outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        loss_all_val.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.6f}")

        # ====== Save best on validation (只在验证集刷新时保存) ======
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_epoch = epoch + 1

            best_name = 'Network_{}_data_{}_best_epoch{}_lr{}_bs{}_{}.pth'.format(
                args.net_version, args.data_version, best_epoch, args.lr, args.BatchSize, args.loss_type
            )
            best_ckpt_path = os.path.join(ckpt_dir, best_name)

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val,
                'config': {
                    'time_num': config.time_num,
                    'patch_size': config.patch_size,
                    'mask_ratio': mask_ratio,
                },
                'args': vars(args) if hasattr(args, '__dict__') else args.__dict__,
            }, best_ckpt_path)
            print(f"[Best Updated] epoch={best_epoch}, val_loss={best_val:.6f}")
            print(f"Saved best checkpoint to {best_ckpt_path}")
            # ====== 可选：保存 last（最后一轮） ======
            if save_last and (epoch + 1 == epochs):
                last_name = 'Network_{}_data_{}_last_epoch{}_lr{}_bs{}_{}.pth'.format(
                    args.net_version, args.data_version, epoch + 1, args.lr, args.BatchSize, args.loss_type
                )
                last_ckpt_path = os.path.join(ckpt_dir, last_name)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val,
                    'best_epoch': best_epoch,
                }, last_ckpt_path)
                print(f"Saved last checkpoint to {last_ckpt_path}")

            # ====== 每个 epoch 同步保存损失曲线（覆盖式） ======
            loss_save_name = 'Loss_results_net_{}_mat_data_{}_lr{}_bs{}_{}.mat'.format(
                args.net_version, args.data_version, args.lr, args.BatchSize, args.loss_type
            )
            loss_save_path = os.path.join(loss_dir, loss_save_name)
            scipy.io.savemat(loss_save_path, {
                'loss_train': loss_all_train,
                'loss_validate': loss_all_val,
                'best_val': best_val,
                'best_epoch': best_epoch
            })
            print(f"Saved loss curves to {loss_save_path}")

        print(f"\nTraining finished. Best Val Loss {best_val:.6f} at epoch {best_epoch}.")
        if best_ckpt_path is not None:
            print(f"Best checkpoint: {best_ckpt_path}")