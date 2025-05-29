import os
os.environ["WANDB_MODE"] = "offline"  # 在导入wandb前设置
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms, models
from tqdm.auto import tqdm
from PIL import Image
import torch.nn as nn

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

#from pix2pix_turbo import Pix2Pix_Turbo
#from pix2pix_turbo_reference import Pix2Pix_Turbo
#from pix2pix_imgEmb import Pix2Pix_Turbo
#from pix2pix_column import Pix2Pix_Turbo
from pix2pix_unliref import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset

in_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    # get the batch_size, depth, height, and width of the Tensor
    b, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(b * d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram 

def get_features(image1, image2, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image1 
    y = image2 
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        y = layer(y)
        if name in layers:
            features[layers[name]] = [x,y]
            
    return features
 
class GradientLoss(nn.Module):
    def __init__(self, operator="Sobel", channel_mean=True):
        r"""
       :param operator: in ['Sobel', 'Prewitt','Roberts','Scharr']
       :param channel_mean: 是否在通道维度上计算均值
       """
        super(GradientLoss, self).__init__()
        assert operator in ['Sobel', 'Prewitt', 'Roberts', 'Scharr'], "Unsupported operator"
        self.channel_mean = channel_mean
        self.operators = {
            "Sobel": {
                'x': torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float)
            },
            "Prewitt": {
                'x': torch.tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]], dtype=torch.float)
            },
            "Roberts": {
                'x': torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[0, -1], [1, 0]]]], dtype=torch.float)
            },
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float)
            },
        }
        self.op_x = self.operators[operator]['x'].cuda()
        self.op_y = self.operators[operator]['y'].cuda()
 
    def gradients(self, img_tensor):
        op_x, op_y = self.op_x, self.op_y
        if self.channel_mean:
            img_tensor = img_tensor.mean(dim=1, keepdim=True)
            groups = 1
        else:
            groups = img_tensor.shape[1]
            op_x = op_x.repeat(groups, 1, 1, 1)
            op_y = op_y.repeat(groups, 1, 1, 1)
        grad_x = F.conv2d(img_tensor, op_x, groups=groups)
        grad_y = F.conv2d(img_tensor, op_y, groups=groups)
        return grad_x, grad_y
 
    def forward(self, img1, img2):
        grad_x1, grad_y1 = self.gradients(img1)
        grad_x2, grad_y2 = self.gradients(img2)
        diff_x = torch.abs(grad_x1 - grad_x2)
        diff_y = torch.abs(grad_y1 - grad_y2)
        total_loss = torch.mean(diff_x + diff_y)
        return total_loss

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
        net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_clip, preprocess = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)

    vgg = models.vgg16(pretrained=True).features
    vgg.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train", tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    vgg = accelerator.prepare(vgg)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)
    vgg.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(os.path.join(args.dataset_folder, "test_B"), model=feat_model, num_workers=0, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)

    # start the training loop
    global_step = 0
    split = nn.Unfold(kernel_size = (8,128),stride=(8,128),padding=0)
    l_g = GradientLoss().cuda()
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # 0513
                #x_src_renorm = (255*x_src[0,:,H//2:,:].permute(1,2,0).cpu().numpy()).astype(np.uint8)
                #x_src_renorm = preprocess(Image.fromarray(x_src_renorm)).unsqueeze(0).cuda()
                #img_embed = net_clip.encode_image(x_src_renorm)
                #img_embed = split(x_src[:,:,H//2:,:]).permute(0,2,1).reshape(B,-1,1024)
                # forward pass
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                #x_tgt_pred = net_pix2pix(x_src, prompt_tokens=img_embed, deterministic=True)
                #x_tgt_pred = x_tgt_pred[:,:,:int(H/2),:]
                loss_vggfeat = 0 
                # 0514 
                # x_tgt_pred_feat = transforms.ToPILImage()(x_tgt_pred[0] * 0.5 + 0.5)
                # x_tgt_pred_feat = get_features(batch["output_feats"]
                #                     , in_transform(x_tgt_pred_feat).unsqueeze(0).cuda(), vgg)
                # for k,v in x_tgt_pred_feat.items(): 
                #     loss_vggfeat += F.mse_loss(gram_matrix(v[0]).float(), gram_matrix(v[1]).float(), reduction="mean") * args.lambda_lpips * 0.5 / 1e8
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                #loss_l2 = torch.abs(x_tgt_pred - x_tgt).mean() * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                if loss_vggfeat > 0: 
                    loss += loss_vggfeat
                # CLIP similarity loss
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim
                accelerator.backward(loss, retain_graph=True)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                lossGD = l_g(x_tgt_pred.float(), x_tgt.float()) * args.lambda_l2 / 4 
                accelerator.backward(lossGD.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                if args.lambda_gan > 0:
                    """
                    Generator loss: fool the discriminator
                    """               
                    x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                    ##x_tgt_pred = x_tgt_pred[:,:,:int(H/2),:]
                    lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Discriminator loss: fake image vs real image
                    """
                    # real image
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    # fake image
                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    if args.lambda_gan > 0:
                        logs["lossG"] = lossG.detach().item()
                        logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["lossGD"] = lossGD.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if loss_vggfeat > 0:
                        logs["loss_vggfeat"] = loss_vggfeat.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            B, C, H, W = x_src.shape
                            #img_embed = split(x_src[:,:,H//2:,:]).permute(0,2,1).reshape(B,-1,1024)
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, prompt_tokens=batch_val["input_ids"].cuda(), deterministic=True)
                                #x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, prompt_tokens=img_embed, deterministic=True)                               
                                #x_tgt_pred = x_tgt_pred[:,:,:int(H/2),:]
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                #loss_l2 = torch.abs(x_tgt_pred - x_tgt).mean() * args.lambda_l2
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                                if args.lambda_clipsim > 0:
                                # compute clip similarity loss
                                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                    caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                    clipsim = clipsim.mean()
                                    l_clipsim.append(clipsim.item())

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())                               
                            # save output images to file for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)
                        if args.track_val_fid:
                            curr_stats = get_folder_features(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        if args.lambda_clipsim > 0:
                            logs["val/clipsim"] = np.mean(l_clipsim)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
