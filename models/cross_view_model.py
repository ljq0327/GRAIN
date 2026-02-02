import torch
from torch import nn
import torch.nn.functional as F
from models.baseline import R3DBackbone


class CrossViewModel(nn.Module):
    def __init__(self, num_actions, prototype_dim=512, momentum=0.95, cross_view_type='CS', view_dropout_rate=0.0, use_view_attention=True):
        super(CrossViewModel, self).__init__()
        self.num_actions = num_actions
        self.cross_view_type = cross_view_type 
        self.view_dropout_rate = view_dropout_rate 
        self.use_view_attention = use_view_attention 
        self.backbone = R3DBackbone() 
        
        self.feature_dim = 512 
        
        # Parameters related to prototype-based contrastive learning
        self.prototype_dim = prototype_dim 
        self.momentum = momentum 
        self.register_buffer('prototypes', torch.randn(num_actions, prototype_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_actions)) 
        self.register_buffer('prototypes_initialized', torch.zeros(num_actions, dtype=torch.bool)) 
        
        # Loss weight parameter
        
        self.consistency_weight = 0.05      # Guidance-Consistency Loss
        self.auxiliary_weight = 0.10        # Auxiliary Classification Loss
        self.view_consistency_weight = 0.25 # cross-view consistency loss
        self.prototype_weight = 0.40        # prototype-based InfoNCE loss
        
        
        self.temperature = 0.07 
        
        # Guided Token Generator
        self.concept_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1)
        )
        
        # Positional Encoding
        self.max_frames = 32  # Maximum supported frame rate
        self.positional_encoding = nn.Parameter(
            torch.zeros(self.max_frames, self.feature_dim), 
            requires_grad=True
        )
        
        # Self-attention module
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.feature_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)
        
        # action classifier
        self.action_classifier = nn.Linear(self.feature_dim, num_actions)  
        self.auxiliary_classifier = nn.Linear(self.feature_dim, num_actions)
        
        # 6. Perspective attention mechanism
        self.view_attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 4, 1) 
        )


    def forward(self, inputs, action_labels=None, return_feats=False, return_attn=False):
        """
            Forward propagation function
            Args:
                Inputs: [batch_size, num_views, 16, 3, 224, 224] Multi-view video data
                action_labels: [batch_size] Action labels, used for calculating the loss
                return_feats: Whether to return features (coarse, refined, fused)
                return_attn: Whether to return attention weights
            Returns:
                action_logits: [batch_size, num_actions]
                consistency_loss: Guidance-Consistency Loss
                view_consistency_loss: cross-view consistency loss
                prototype_loss: prototype-based InfoNCE loss
                auxiliary_loss: Auxiliary Classification Loss
            """
        batch_size, num_views = inputs.shape[0], inputs.shape[1]
        target_views = num_views 
        
        all_raw_features = []  
        all_coarse_concepts = [] 
        
        for view_idx in range(target_views):
            view_input = inputs[:, view_idx, :, :, :, :]  # [batch_size, 16, 3, 224, 224]
            view_input = view_input.permute(0, 2, 1, 3, 4)  # [batch_size, 3, 16, 224, 224]
            view_raw_features = self.backbone(view_input)  # [batch_size, 512, T, 1, 1]
            view_raw_features = view_raw_features.squeeze(-1).squeeze(-1)  # [batch_size, 512, T]
            view_raw_features = view_raw_features.permute(0, 2, 1)  # [batch_size, T, 512]
            
            all_raw_features.append(view_raw_features)

            # Extracting action information
            time_dim = view_raw_features.shape[1]  
            view_raw_reshaped = view_raw_features.reshape(batch_size * time_dim, -1)  # [batch_size*T, 512]
            view_coarse_concepts = self.concept_layer(view_raw_reshaped)  # [batch_size*T, 512]
            view_coarse_concepts = view_coarse_concepts.reshape(batch_size, time_dim, -1)  # [batch_size, T, 512]
            
            # Add positional encoding (dynamically adapting to the time dimension)
            if time_dim <= self.max_frames:
                view_coarse_concepts = view_coarse_concepts + self.positional_encoding[:time_dim].unsqueeze(0)
            else:
                pos_encoding = F.interpolate(
                    self.positional_encoding.unsqueeze(0).transpose(1, 2),  # [1, 512, max_frames]
                    size=time_dim, mode='linear', align_corners=False
                ).transpose(1, 2)  # [1, time_dim, 512]
                view_coarse_concepts = view_coarse_concepts + pos_encoding
            
            all_coarse_concepts.append(view_coarse_concepts)
        
        all_raw_features = torch.stack(all_raw_features, dim=1)      # [batch_size, num_views, T, 512]
        all_coarse_concepts = torch.stack(all_coarse_concepts, dim=1) # [batch_size, num_views, T, 512]
        
        r3d_raw_per_view = all_raw_features.mean(dim=2)  # [batch_size, num_views, 512] 
        
        all_refined_concepts = []
        for view_idx in range(target_views):
            current_coarse_concepts = all_coarse_concepts[:, view_idx, :, :]  # [batch_size, T, 512]
            current_raw_features = all_raw_features[:, view_idx, :, :]
            time_dim = current_coarse_concepts.shape[1]

            # Detailed representation
            attn_output, _ = self.self_attention(
                current_coarse_concepts, current_raw_features, current_raw_features
            )  # [batch_size, T, 512]
        
            current_coarse_concepts = self.norm1(current_coarse_concepts + attn_output)

            ff_output = self.feed_forward(current_coarse_concepts)  # [batch_size, T, 512]
            
            view_refined_concepts = self.norm2(current_coarse_concepts + ff_output)  # [batch_size, T, 512]
            
            view_refined_concepts = view_refined_concepts.mean(dim=1)  # [batch_size, 512]
            
            all_refined_concepts.append(view_refined_concepts)
        
        
        refined_concepts = torch.stack(all_refined_concepts, dim=1)  # [batch_size, target_views, 512]
        coarse_concepts = all_coarse_concepts.mean(dim=2)  # [batch_size, target_views, 512]
        
        if action_labels is not None:
            view_consistency_loss = self.compute_view_consistency_loss(refined_concepts, action_labels)
            auxiliary_loss = self.compute_auxiliary_classification_loss(refined_concepts, action_labels)
            consistency_loss = self.compute_consistency_loss(coarse_concepts, refined_concepts)
        else:
            view_consistency_loss = torch.tensor(0.0, device=refined_concepts.device)
            auxiliary_loss = torch.tensor(0.0, device=refined_concepts.device)
            consistency_loss = torch.tensor(0.0, device=refined_concepts.device)

        drop = None 
        if self.training and self.view_dropout_rate > 0.0 and self.cross_view_type == 'CV':
            B, V, D = refined_concepts.shape
            drop = (torch.rand(B, V, device=refined_concepts.device) < self.view_dropout_rate)
            all_dropped = drop.all(dim=1)
            if all_dropped.any():
                rows = torch.nonzero(all_dropped, as_tuple=False).squeeze(1)
                keep_idx = torch.randint(0, V, (rows.numel(),), device=drop.device)
                drop[rows, :] = True
                drop[rows, keep_idx] = False
            refined_concepts = refined_concepts.masked_fill(drop.unsqueeze(-1), 0.0)
        
        if self.use_view_attention:
            view_logits = self.view_attention(refined_concepts)          # [B, V, 1]
            if drop is not None:
                masked_logits = view_logits.squeeze(-1).masked_fill(drop, float('-inf'))
                attn = torch.softmax(masked_logits, dim=1).unsqueeze(-1)  # [B, V, 1]
            else:
                attn = torch.softmax(view_logits.squeeze(-1), dim=1).unsqueeze(-1)  # [B, V, 1]
            fused_concepts = (refined_concepts * attn).sum(dim=1)        # [B, 512]
        else:
            if drop is not None:
                valid_views = (~drop).sum(dim=1, keepdim=True).float()  # [B, 1]
                masked_concepts = refined_concepts.masked_fill(drop.unsqueeze(-1), 0.0)
                fused_concepts = masked_concepts.sum(dim=1) / valid_views.squeeze(-1).clamp_min(1)  # [B, 512]
            else:
                fused_concepts = refined_concepts.sum(dim=1)  # [batch_size, 512]
        
        action_logits = self.action_classifier(fused_concepts)  # [batch_size, num_actions]
        
        if action_labels is not None:
            prototype_loss = self.compute_prototype_contrastive_loss(fused_concepts, action_labels)
        else:
            prototype_loss = torch.tensor(0.0, device=fused_concepts.device)
        

        # Used when saving offline features, for visualization and comparison.
        extras = {}
        if return_feats:
            if self.use_view_attention:
                r3d_view_logits = self.view_attention(r3d_raw_per_view) 
                if drop is not None:
                    r3d_masked_logits = r3d_view_logits.squeeze(-1).masked_fill(drop, float('-inf'))
                    r3d_attn = torch.softmax(r3d_masked_logits, dim=1).unsqueeze(-1) 
                else:
                    r3d_attn = torch.softmax(r3d_view_logits.squeeze(-1), dim=1).unsqueeze(-1) 
                r3d_fused = (r3d_raw_per_view * r3d_attn).sum(dim=1) 
            else:
                if drop is not None:
                    valid_views = (~drop).sum(dim=1, keepdim=True).float()
                    masked_r3d = r3d_raw_per_view.masked_fill(drop.unsqueeze(-1), 0.0)
                    r3d_fused = masked_r3d.sum(dim=1) / valid_views.squeeze(-1).clamp_min(1) 
                else:
                    r3d_fused = r3d_raw_per_view.sum(dim=1) 
            
            extras.update({
                "r3d_raw":  r3d_raw_per_view,  
                "r3d_fused": r3d_fused,  
                "coarse":  coarse_concepts,     
                "refined": refined_concepts,   
                "fused":   fused_concepts,      
            })
        if return_attn:
            if self.use_view_attention:
                extras.update({
                    "attn": attn.squeeze(-1), 
                    "view_logits": view_logits.squeeze(-1), 
                })
            else:
                extras.update({
                    "attn": None,
                    "view_logits": None,
                })

        if return_feats or return_attn:
            return action_logits, consistency_loss, view_consistency_loss, prototype_loss, auxiliary_loss, extras
        return action_logits, consistency_loss, view_consistency_loss, prototype_loss, auxiliary_loss

    def compute_consistency_loss(self, coarse_concepts, refined_concepts):

        coarse_norm = torch.nn.functional.normalize(coarse_concepts, p=2, dim=-1)

        refined_norm = torch.nn.functional.normalize(refined_concepts, p=2, dim=-1)
        
        similarity = torch.sum(coarse_norm * refined_norm, dim=-1) 
        
        consistency_loss = torch.mean((similarity - 1.0) ** 2)
        
        return consistency_loss
    
    def compute_prototype_contrastive_loss(self, fused_features, action_labels, top_k_negatives=None):
        device = fused_features.device
        B, Df = fused_features.shape
        C, Dp = self.prototypes.shape

        if Df != Dp:
            if not hasattr(self, 'proto_proj'):
                self.proto_proj = nn.Linear(Df, Dp).to(device)
            feats = self.proto_proj(fused_features)
        else:
            feats = fused_features
        feats = F.normalize(feats, p=2, dim=1)                     

        with torch.no_grad():
            proto_snap = self.prototypes.detach().clone() 
            valid_mask = self.prototypes_initialized if hasattr(self,'prototypes_initialized') \
                         else torch.ones(C, dtype=torch.bool, device=device)
        proto_snap = F.normalize(proto_snap, p=2, dim=1)
        logits_all = feats @ proto_snap.t() / self.temperature         

        neg_mask = valid_mask.unsqueeze(0).expand(B, C) 
        masked_logits = logits_all.masked_fill(~neg_mask, float('-inf'))

        pos_idx = action_labels.view(-1)
        has_pos = valid_mask[pos_idx]

        if has_pos.any():
            logits = masked_logits[has_pos]           
            pos    = pos_idx[has_pos]     
            if top_k_negatives is not None:
                b = torch.arange(logits.size(0), device=device)
                valid_neg = neg_mask[has_pos].clone()
                valid_neg[b, pos] = False
                num_neg = valid_neg.sum(dim=1)
                k = min(int(top_k_negatives), int(num_neg.max().item()) if num_neg.numel() else 0)
                if k > 0:
                    logits_neg_only = logits.masked_fill(~valid_neg, float('-inf'))
                    _, topk_idx = torch.topk(logits_neg_only, k=k, dim=1)
                    keep = torch.zeros_like(valid_neg)
                    keep[b, pos] = True
                    keep[b.unsqueeze(1), topk_idx] = True
                    logits = logits.masked_fill(~keep, float('-inf'))
                else:
                    keep = F.one_hot(pos, num_classes=logits.size(1)).bool()
                    logits = logits.masked_fill(~keep, float('-inf'))
            log_denom = torch.logsumexp(logits, dim=1)
            pos_logit = logits.gather(1, pos.view(-1,1)).squeeze(1)
            loss = (-(pos_logit - log_denom)).mean()
        else:
            loss = fused_features.new_tensor(0.0)

        with torch.no_grad():
            import torch.distributed as dist
            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                self.update_prototypes(feats.detach(), action_labels)
        return loss
    
    def compute_view_consistency_loss(self, refined_concepts, action_labels):
        
        B, V, D = refined_concepts.shape
        device = refined_concepts.device
        N = B * V

        feats  = F.normalize(refined_concepts.reshape(N, D), p=2, dim=-1)
        labels = action_labels.view(B, 1).expand(B, V).reshape(N)
        inst_ids = torch.arange(B, device=device).view(B, 1).expand(B, V).reshape(N)

        logits   = feats @ feats.t() / self.temperature 
        not_self = ~torch.eye(N, dtype=torch.bool, device=device)

        pos_mask = (inst_ids[:, None] == inst_ids[None, :]) & not_self     # The same instance
        neg_mask = (labels[:, None]    != labels[None, :]) & not_self      # Different actions
        valid    = pos_mask | neg_mask

        masked   = logits.masked_fill(~valid, float('-inf'))
        row_ok   = valid.any(dim=1)
        safe     = torch.where(row_ok.unsqueeze(1), masked, logits)
        log_Z    = torch.logsumexp(safe, dim=1, keepdim=True)
        log_prob = logits - log_Z

        pos_cnt  = pos_mask.sum(dim=1).clamp(min=1)
        mean_log_pos = (pos_mask * log_prob).sum(dim=1) / pos_cnt
        return -mean_log_pos.mean()
    
    def compute_auxiliary_classification_loss(self, refined_concepts, action_labels):

        batch_size, num_views, feature_dim = refined_concepts.shape
        
        auxiliary_loss = 0.0
        
        for view_idx in range(num_views):
            view_features = refined_concepts[:, view_idx, :]
    
            view_logits = self.auxiliary_classifier(view_features)  # [batch_size, num_actions]
            
            view_loss = F.cross_entropy(view_logits, action_labels)
            auxiliary_loss += view_loss
        
        return auxiliary_loss / num_views
    
    def update_prototypes(self, fused_features, action_labels):
        with torch.no_grad():
            device = fused_features.device
            C, D = self.prototypes.shape

            if not hasattr(self, 'prototypes_initialized'):
                self.prototypes_initialized = torch.zeros(C, dtype=torch.bool, device=device)

            for cls in torch.unique(action_labels):
                cls = int(cls.item()) if cls.dtype != torch.int64 else int(cls)
                cls_mask = (action_labels == cls)
                if cls_mask.any():
                    cls_mean = fused_features[cls_mask].mean(dim=0) 
                    if not self.prototypes_initialized[cls]:
                        self.prototypes[cls] = cls_mean
                        self.prototypes_initialized[cls] = True
                    else:
                        m = float(self.momentum)
                        self.prototypes[cls] = m * self.prototypes[cls] + (1 - m) * cls_mean

                    if hasattr(self, 'prototype_counts'):
                        self.prototype_counts[cls] += int(cls_mask.sum().item())