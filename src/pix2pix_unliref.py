import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim = 32, bias_bool = False):
        super(SelfAttention, self).__init__()

        # Query, Key, Value参数矩阵
        self.query_matrix = nn.Linear(input_dim, hidden_dim, bias=bias_bool)
        self.key_matrix = nn.Linear(input_dim, hidden_dim, bias=bias_bool)
        self.value_matrix = nn.Linear(input_dim, hidden_dim, bias=bias_bool)
        
        # Dropout层
        self.dropout = nn.Dropout(0.0)

        # 注意力分数归一化的比例系数
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).cuda()
        
        self.linear = nn.Linear(hidden_dim, input_dim, bias=bias_bool)

    def forward(self, x, y):
        """
        x: 输入的特征矩阵，维度为(batch_size, seq_len, hidden_dim)
        """
        # 获取batch_size和seq_len
        batch_size, seq_len, hidden_dim = x.size()

        # 得到Query, Key, Value
        Q = self.query_matrix(y)  # (batch_size, seq_len, hidden_dim)
        K = self.key_matrix(x)  # (batch_size, seq_len, hidden_dim)
        V = self.value_matrix(x)  # (batch_size, seq_len, hidden_dim)

        # 计算注意力分数（内积）
        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行缩放
        scaled_scores = scores / self.scale  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行softmax，得到注意力权重
        attn_weights = torch.softmax(scaled_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # 对注意力权重进行dropout
        attn_weights = self.dropout(attn_weights)

        # 将注意力权重与Value相乘，得到self-attention后的表示
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        attn_output = self.linear(attn_output)  # (batch_size, seq_len, input_dim)

        batch_size, seq_len, input_dim = attn_output.shape

        return attn_output

class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        self.attention1 = SelfAttention(4,bias_bool = True).cuda()

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            self.attention1.load_state_dict(sd["attention1"])

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([299], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.attention1.eval()
        self.attention1.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        self.attention1.train()

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
            #caption_enc = prompt_tokens
        
        if deterministic:
            _, _, H, _ = c_t.shape
            encoded_control1 = self.vae.encode(c_t[:,:,:H//3,:]).latent_dist.sample() * self.vae.config.scaling_factor 
            encoded_control2 = self.vae.encode(c_t[:,:,H//3:H*2//3,:]).latent_dist.sample() * self.vae.config.scaling_factor
            _, _, h, _ = encoded_control1.shape
            encoded_control1[:,:,h*3//5:,:] = 0
            encoded_control2[:,:,:h*3//5,:] = 0   
            encoded_control = encoded_control1 + encoded_control2
            encoded_ref = self.vae.encode(c_t[:,:,H*2//3:,:]).latent_dist.sample() * self.vae.config.scaling_factor
            B, C, H, W = encoded_ref.shape
            encoded_att = encoded_ref.reshape(B,C,-1).permute(0,2,1) # BCHW->BC(HW)->B(HW)C
            encoded_att = self.attention1(encoded_att, encoded_control.reshape(B,C,-1).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)
            #encoded_att = self.attention1(encoded_att, encoded_att).permute(0,2,1).reshape(B,C,H,W)
            model_pred = self.unet(encoded_att+encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        sd["attention1"] = self.attention1.state_dict()
        torch.save(sd, outf)
