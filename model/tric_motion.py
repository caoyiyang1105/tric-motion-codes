import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder import load_bert
from utils.misc import WeightedSum

from model.modules import TriCModel
from einops import rearrange, repeat
import math


class TriCMotion(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=512, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='tri_domain', emb_trans_dec=False, n_frames=196, is_training=False, ds_ratio=4, **kargs):
        super().__init__()

        self.is_training = is_training
        print('is_training: ', self.is_training)

        self.n_frames = n_frames

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.ds_ratio = ds_ratio
        self.input_feats = 12*self.ds_ratio

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.mask_frames = kargs.get('mask_frames', False)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(
            self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.emb_policy = kargs.get('emb_policy', 'add')

        self.sequence_pos_encoder = PositionalEncoding2D(
            self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.pred_len = kargs.get('pred_len', 0)
        self.context_len = kargs.get('context_len', 0)
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.total_len > 0
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])


        print("TRI_DOMAIN init")
        self.tric_model = TriCModel(
            d_model=self.latent_dim, nlayers=num_layers, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, is_training=self.is_training, ds_ratio=self.ds_ratio)

        self.embed_timestep = TimestepEmbedder(self.latent_dim)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')

                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')

                if self.text_encoder_type == "clip":
                    print('Loading CLIP...')
                    self.clip_version = clip_version
                    self.clip_model = self.load_and_freeze_clip(clip_version)
                    self.encode_text = self.clip_encode_text
                elif self.text_encoder_type == 'bert':
                    print("Loading BERT...")
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'
                    bert_model_path = 'model/BERT/distilbert-base-uncased'
                    # Sorry for that, the naming is for backward compatibility
                    self.clip_model = load_bert(bert_model_path)
                    self.encode_text = self.bert_encode_text
                    self.clip_dim = 768
                else:
                    raise ValueError(
                        'We only support [CLIP, BERT] text encoders')

                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(
                    self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        word_embed, sentence_embed = cond
        seq_len, bs, d = word_embed.shape
        # [38, 64, 768]
        # print('word_embed.shape', word_embed.shape)
        # [64, 768]
        # print('sentence_embed.shape', sentence_embed.shape)
        if force_mask:
            word_embed = torch.zeros_like(word_embed)
            sentence_embed = torch.zeros_like(sentence_embed)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(
                bs, device=word_embed.device) * self.cond_mask_prob).view(1, bs, 1)
            word_embed = word_embed * (1. - mask)
            sentence_embed = sentence_embed * (1. - mask)
        return word_embed, sentence_embed

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        # Specific hardcoding for humanml dataset
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(
                device)  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length -
                                   context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            # [bs, context_length] # if n_tokens > 77 -> will truncate
            texts = clip.tokenize(raw_text, truncate=True).to(device)
        return self.clip_model.encode_text(texts).float().unsqueeze(0)

    def bert_encode_text(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text, mask = self.clip_model(raw_text)

        return self.create_text_embed(enc_text, mask)

    def create_text_embed(self, enc_text, mask):
        enc_text = enc_text.permute(1, 0, 2)
        sentence_embed = enc_text[0, :, :]  # [bs, 768]

        word_embed = enc_text[1:, :, :]  # [seq_len, bs, 768]

        text_mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        text_mask = text_mask[:, 1:]
        return word_embed, sentence_embed, text_mask


    def whole2joint(self, x):
        # x: [bs, 263, 1, 196] -> [bs, 196, 22, 12]
        B, D0, _, T0 = x.shape

        motion = rearrange(x, 'b d 1 t -> (b t) d')
        n_j = 22
        if motion.shape[1] == 263:
            n_j = 22
        elif motion.shape[1] == 251:
            n_j = 21
        n_feat = 12
        BT = motion.shape[0]
        x2 = motion[:, 4: 4 + (n_j - 1) * 3]
        x3 = motion[:, 4 + (n_j - 1) * 3: 4 + (n_j - 1) * 9]
        x4 = motion[:, 4 + (n_j - 1) * 9: 4 + (n_j - 1) * 9 + n_j * 3]
        x_pos = x2.reshape(BT, n_j-1, 3)
        x_rot = x3.reshape(BT, n_j-1, 6)
        x_speed = x4.reshape(BT, n_j, 3)
        x_joints = torch.zeros(
            [BT, n_j, n_feat], dtype=torch.float32, device=x.device)
        x_joints[:, 1:, :3] = x_pos
        x_joints[:, 1:, 3:9] = x_rot
        x_joints[:, :, 9:12] = x_speed

        x_root = torch.cat([motion[:, :4], motion[:, -4:]], dim=1)
        x_joints[:, 0, :8] = x_root

        x = x_joints
        x = rearrange(x, '(b t) j d -> b t j d', b=B, t=T0)

        return x

    def joint2whole(self, x):
        B, T0, n_j, _ = x.shape
        # [bs, 196, 22, 12] -> [bs, 263, 1, 196]j
        x = rearrange(x, 'b t j d -> (b t) j d')
        if n_j==22:
            jmotion = torch.zeros([B, T0, 263], device=x.device) 
        elif n_j==21:
            jmotion = torch.zeros([B, T0, 251], device=x.device)
        # root
        jmotion[:, :, :4] = x[:, 0, :4].reshape(B, T0, -1)
        # fc
        jmotion[:, :, -4:] = x[:, 0, 4:8].reshape(B, T0, -1)
        # x_pos
        jmotion[:, :, 4: 4 + (n_j - 1) * 3] = x[:,
                                                1:, :3].reshape(B, T0, -1)
        # x_rot
        jmotion[:, :, 4 + (n_j - 1) * 3: 4 + (n_j - 1)
                * 9] = x[:, 1:, 3:9].reshape(B, T0, -1)
        # x_speed
        jmotion[:, :, 4 + (n_j - 1) * 9: 4 + (n_j - 1)
                * 9 + n_j * 3] = x[:, :, 9:12].reshape(B, T0, -1)

        jmotion = jmotion.permute(0, 2, 1).unsqueeze(2)

        return jmotion

    def forward(self, x, timesteps, y=None, is_training=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        B, D0, _, T0 = x.shape

        x = self.whole2joint(x)
        n_j = 22

        # t, j, b, d
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]
        time_emb = rearrange(time_emb, 't b d -> b t d')  # [bs, 1, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            elif 'text_bert' in y.keys():
                enc_text = self.create_text_embed(
                    y['text_bert'], y['text_mask'])
            else:
                enc_text = self.encode_text(y['text'])

            if type(enc_text) == tuple:
                word_embed, sentence_embed, text_mask = enc_text

                if text_mask.shape[0] == 1 and B > 1:
                    text_mask = torch.repeat_interleave(text_mask, B, dim=0)

            word_embed, sentence_embed = self.mask_cond(
                (word_embed, sentence_embed), force_mask=force_mask)
            word_embed = self.embed_text(word_embed)
            sentence_embed = self.embed_text(sentence_embed)

            if self.emb_policy == 'add':
                word_embed = time_emb + word_embed.permute(1, 0, 2)

        B, T, n_j, n_feat = x.shape
        x = rearrange(x, 'b (t p) j d -> b t j (d p)', p=self.ds_ratio)
        x = self.input_process(x)

        frames_mask = None
        # Don't use mask with the generate script
        is_valid_mask = y['mask'].shape[-1] > 1
        if self.mask_frames and is_valid_mask:
            frames_mask = torch.logical_not(
                # y['mask'][..., :x.shape[1]].squeeze(1).squeeze(1)).to(device=x.device)
                y['mask'][..., :T].squeeze(1).squeeze(1)).to(device=x.device)
            frames_mask = rearrange(frames_mask, 'b (t p) -> b t p', p=self.ds_ratio)
            frames_mask = frames_mask.all(dim=2)

            if self.emb_trans_dec or self.arch == 'trans_enc':
                step_mask = torch.zeros(
                    (B, 1), dtype=torch.bool, device=x.device)
                frames_mask = torch.cat([step_mask, frames_mask], dim=1)


        time_emb = repeat(time_emb, 'b t d -> b t j d',
                          j=n_j)  # [bs, 196, 22, d]
        xseq = torch.cat((time_emb, x), axis=1)
        # [t, j, b, d]
        xseq = rearrange(xseq, 'b t j d -> t j b d')
        xseq = self.sequence_pos_encoder(xseq)
        xseq = rearrange(xseq, 't j b d -> b t j d')

        # forward
        x, casual_lst = self.tric_model(xseq, frames_mask=frames_mask, sentence_embed=sentence_embed, word_embed=word_embed, text_mask=text_mask)

        x = x[:, 1:]
        x = self.output_process(x)
        x = rearrange(x, 'b t j (d p) -> b (t p) j d', p=self.ds_ratio)

        x = self.joint2whole(x)

        # [bs, 263, 1, 196]
        if is_training:
            if len(casual_lst) ==1:
                x_a, x_f, x_cf = casual_lst[0]
                x_f1 = self.output_process(x_f)
                x_cf1 = self.output_process(x_cf)
                tde = x_f1- x_cf1
                casual_lst[0] = (x_a, x_f, x_cf, tde, x_f1)
            return x, casual_lst
        else:
            return x


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout=0.1, height=200, width=50):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(
            pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(
            pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(
            pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.permute(1, 2, 0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [t, j, b, d]
        x = x + self.pe[:x.shape[0], :x.shape[1], None, :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = PositionalEncoding(latent_dim)

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, njoints, nfeats = x.shape

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        bs, nframes, njoints, dim = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


class EmbedTargetLocSingle(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + \
            ['traj', 'heading']
        self.target_cond_dim = len(
            self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)
        self.latent_dim = latent_dim
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]
        for _ in range(num_layers):
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]
        self.mlp = nn.Sequential(*_layers)

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(
                sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx,
                         self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-
                              1).view(input.shape[0], -1)
        return self.mlp(mlp_input)


class EmbedTargetLocSplit(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + \
            ['traj', 'heading']
        self.target_cond_dim = 4
        self.latent_dim = latent_dim
        self.splited_dim = self.latent_dim // len(
            self.extended_goal_joint_names)
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0
        self.mini_mlps = nn.ModuleList()
        for _ in self.extended_goal_joint_names:
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]
            for _ in range(num_layers):
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim,
                                                 self.splited_dim)]
            self.mini_mlps.append(nn.Sequential(*_layers))

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(
                sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx,
                         self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1)
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i])
                      for i in range(mlp_input.shape[1])]
        return torch.cat(mlp_splits, dim=-1)


class EmbedTargetLocMulti(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim):
        super().__init__()

        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions
        self.extended_goal_joint_names = all_goal_joint_names + \
            ['traj', 'heading']
        self.extended_goal_joint_idx = {
            joint_name: idx for idx, joint_name in enumerate(self.extended_goal_joint_names)}
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)
        self.target_loc_emb = nn.ParameterDict({joint_name:
                                                nn.Sequential(
                                                    nn.Linear(3, latent_dim),
                                                    nn.SiLU(),
                                                    nn.Linear(latent_dim, latent_dim))
                                                for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        # nn.Linear(self.n_extended_goal_joints, latent_dim)
        self.target_all_loc_emb = WeightedSum(self.n_extended_goal_joints)
        self.latent_dim = latent_dim

    def forward(self, input, target_joint_names, target_heading):
        output = torch.zeros(
            (input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)

        # Iterate over the batch and apply the appropriate filter for each joint
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(
                sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            output_one_sample = torch.zeros(
                (self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype, device=input.device)
            for joint_name in sample_joint_names_w_heading:
                layer = self.target_loc_emb[joint_name]
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(
                    input[sample_idx, self.extended_goal_joint_idx[joint_name]])
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())

        return output
