import copy
import os
import warnings
import random

warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
)
from torchvision.transforms import (
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
)

from dataloader import CrossViewDataLoader, cross_view_collate
from models.cross_view_model import CrossViewModel


def train_cross_view_epoch(epoch, data_loader, model, optimizer, criterion, writer, use_cuda, args):
    """
    Cross-view training function

    Processes multi-view video data using the following loss functions:
    - Classification loss: for the main classification task
    - Consistency loss: for coarse-to-fine concept consistency
    - View consistency loss: consistency constraint between different views
    - Prototype contrastive loss: to enhance action discriminability
    - Auxiliary classification loss: for single-view auxiliary classification
    """
    print(f'train cross-view at epoch {epoch}', flush=True)

    model.train()
    

    m = model.module if hasattr(model, 'module') else model
    if epoch < args.dropout_switch_epoch:
        m.view_dropout_rate = args.early_dropout_rate
        dropout_info = f"Early dropout rate: {args.early_dropout_rate}"
    else:
        m.view_dropout_rate = args.late_dropout_rate
        dropout_info = f"Post-training dropout rate: {args.late_dropout_rate}"
    
    print(f"Epoch {epoch}: {dropout_info}")
    
    total_loss = 0
    total_consistency_loss = 0
    total_view_consistency_loss = 0
    total_prototype_loss = 0
    total_auxiliary_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, (clips, viewpoints, actions, batch_indices) in enumerate(tqdm(data_loader)):
        clips = Variable(clips.type(torch.FloatTensor)).cuda()
        viewpoints = Variable(viewpoints.type(torch.LongTensor)).cuda()
        actions = Variable(actions.type(torch.LongTensor)).cuda()

        optimizer.zero_grad()

        # Use the action from the first-person perspective as the label.
        action_labels = actions[:, 0]

        action_logits, consistency_loss, view_consistency_loss, prototype_loss, auxiliary_loss = model(clips, action_labels)

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        classification_loss = criterion(action_logits, action_labels)

        if hasattr(model, 'module'):
            consistency_weight = model.module.consistency_weight
            view_consistency_weight = model.module.view_consistency_weight
            prototype_weight = model.module.prototype_weight
            auxiliary_weight = model.module.auxiliary_weight
        else:
            consistency_weight = model.consistency_weight
            view_consistency_weight = model.view_consistency_weight
            prototype_weight = model.prototype_weight
            auxiliary_weight = model.auxiliary_weight

        total_loss_batch = (classification_loss + 
                           consistency_weight * consistency_loss + 
                           view_consistency_weight * view_consistency_loss + 
                           prototype_weight * prototype_loss + 
                           auxiliary_weight * auxiliary_loss)
        total_loss_batch = total_loss_batch.mean()  

        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item() if total_loss_batch.numel() == 1 else total_loss_batch.mean().item()
        total_consistency_loss += consistency_loss.item() if consistency_loss.numel() == 1 else consistency_loss.mean().item()
        total_view_consistency_loss += view_consistency_loss.item() if view_consistency_loss.numel() == 1 else view_consistency_loss.mean().item()
        total_prototype_loss += prototype_loss.item() if prototype_loss.numel() == 1 else prototype_loss.mean().item()
        total_auxiliary_loss += auxiliary_loss.item() if auxiliary_loss.numel() == 1 else auxiliary_loss.mean().item()

        _, predicted = torch.max(action_logits, 1)
        correct_predictions += (predicted == action_labels).sum().item()
        total_predictions += action_labels.size(0)

    avg_loss = total_loss / len(data_loader)
    avg_consistency_loss = total_consistency_loss / len(data_loader)
    avg_view_consistency_loss = total_view_consistency_loss / len(data_loader)
    avg_prototype_loss = total_prototype_loss / len(data_loader)
    avg_auxiliary_loss = total_auxiliary_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions

    print(f'Epoch {epoch} - Loss: {avg_loss:.4f}, Consistency Loss: {avg_consistency_loss:.4f}, View Consistency Loss: {avg_view_consistency_loss:.4f}, Prototype Loss: {avg_prototype_loss:.4f}, Auxiliary Loss: {avg_auxiliary_loss:.4f}, Accuracy: {accuracy:.4f}')

    return avg_loss, accuracy

def val_cross_view_multi_view_epoch(cfg, epoch, data_loader, model, writer, use_cuda, args):
    """
    Cross-View Multi-View Validation Function

    Direct validation using multi-view data:
    - Input: Multi-view video data [batch_size, num_views, frames, channels, height, width]
    - Processing: Cross-view interaction and feature fusion
    - Output: Action classification prediction
    """
    print(f'validation multi-view at epoch {epoch}')
    model.eval()

    total_loss = 0
    total_consistency_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i, (clips, viewpoints, actions, batch_indices) in enumerate(tqdm(data_loader)):
            clips = Variable(clips.type(torch.FloatTensor)).cuda()
            viewpoints = Variable(viewpoints.type(torch.LongTensor)).cuda()
            actions = Variable(actions.type(torch.LongTensor)).cuda()

            # Use the action from the first-person perspective as the label.
            action_labels = actions[:, 0]
            
            action_logits, consistency_loss, view_consistency_loss, prototype_loss, auxiliary_loss = model(clips, action_labels)

            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            classification_loss = criterion(action_logits, action_labels)

            if hasattr(model, 'module'):
                consistency_weight = model.module.consistency_weight
            else:
                consistency_weight = model.consistency_weight
            
            total_loss_batch = classification_loss + consistency_weight * consistency_loss

            total_loss += total_loss_batch.item() if total_loss_batch.numel() == 1 else total_loss_batch.mean().item()
            total_consistency_loss += consistency_loss.item() if consistency_loss.numel() == 1 else consistency_loss.mean().item()

            _, predicted = torch.max(action_logits, 1)
            correct_predictions += (predicted == action_labels).sum().item()
            total_predictions += action_labels.size(0)

    if len(data_loader) == 0:
        print('The validation data loader is empty, skipping validation.')
        return 0.0
    
    avg_loss = total_loss / len(data_loader)
    avg_consistency_loss = total_consistency_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions

    print(
        f'Validation Epoch {epoch} - Loss: {avg_loss:.4f}, Consistency Loss: {avg_consistency_loss:.4f}, Accuracy: {accuracy:.4f}')

    # Record to TensorBoard
    if writer:
        writer.add_scalar(f'MultiView_Validation/Loss', avg_loss, epoch)
        writer.add_scalar(f'MultiView_Validation/Consistency_Loss', avg_consistency_loss, epoch)
        writer.add_scalar(f'MultiView_Validation/Accuracy', accuracy, epoch)

    return accuracy


def train_model(cfg, run_id, save_dir, use_cuda, args, writer):
    shuffle = True
    print("Run ID : " + args.run_id)

    print("Parameters used : ")
    print("batch_size: " + str(args.batch_size))
    print("lr: " + str(args.learning_rate))

    transform_train = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            RandomShortSideScale(
                min_size=224,
                max_size=256,
            ),
            RandomCrop(224),
            RandomHorizontalFlip(p=0.5)

        ]
    )
    transform_test = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(
                size=256
            ),
            CenterCrop(224)
        ]
    )

    if hasattr(args, 'training_mode') and args.training_mode in ['CS', 'CV']:
        training_mode = args.training_mode
        cfg.training_mode = training_mode  
        print(f"Using {training_mode} Evaluation Agreement")

        train_data_gen = CrossViewDataLoader(cfg, 'train', cross_view_type=training_mode, transform=transform_train)

        train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle,
                                      num_workers=args.num_workers, drop_last=True, 
                                      collate_fn=lambda batch: cross_view_collate(batch, training_mode))

        validation_mode = getattr(args, 'validation_mode', 'both')
        val_dataloader = None
        single_view_dataloader = None
        
        if validation_mode in ['multi_view', 'both']:
            if training_mode == 'CV':
                val_cross_view_type = 'CV_test'
            else:
                val_cross_view_type = 'CS'
            
            val_data_gen = CrossViewDataLoader(cfg, 'test', cross_view_type=val_cross_view_type, transform=transform_test)
            val_dataloader = DataLoader(val_data_gen, batch_size=args.batch_size, shuffle=shuffle,
                                        num_workers=args.num_workers, drop_last=False, 
                                        collate_fn=lambda batch: cross_view_collate(batch, val_cross_view_type))
        
        if validation_mode in ['single_view', 'both']:
            if training_mode == 'CV':
                single_view_type = 'CV_test'  
            elif training_mode == 'CS':
                single_view_type = 'cs_single_test' 
            else:
                single_view_type = None  
            
            if single_view_type is not None:
                single_view_data_gen = CrossViewDataLoader(cfg, 'test', cross_view_type=single_view_type, transform=transform_test)
                single_view_dataloader = DataLoader(single_view_data_gen, batch_size=args.batch_size, shuffle=shuffle,
                                                   num_workers=args.num_workers, drop_last=False, 
                                                   collate_fn=lambda batch: cross_view_collate(batch, single_view_type))
            else:
                single_view_dataloader = None

        view_dropout_rate = getattr(args, 'view_dropout_rate', 0.0)  
        use_view_attention = getattr(args, 'use_view_attention', True) 
        
        
        model_cross_view_type = training_mode
        
        model = CrossViewModel(cfg.num_actions, cross_view_type=model_cross_view_type, 
                             view_dropout_rate=view_dropout_rate, use_view_attention=use_view_attention)

        num_gpus = len(args.gpu.split(','))
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
        model.cuda()
        
        m = model.module if hasattr(model, 'module') else model
        if args.training_mode == 'CV':
            m.view_consistency_weight = 0.25
            m.prototype_weight        = 0.40
            m.auxiliary_weight        = 0.10
            m.temperature             = 0.07

        if args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                        weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        criterion = torch.nn.CrossEntropyLoss()

        start_epoch = 0
        if hasattr(args, 'checkpoint') and args.checkpoint:
            print(f"Restore from checkpoint: {args.checkpoint}")
            if os.path.exists(args.checkpoint):
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    start_epoch = checkpoint.get('epoch', 0)
                    print(f"Model weights loaded successfully, starting from epoch {start_epoch}.")
                else:
                    model.load_state_dict(checkpoint)
                    start_epoch = 0
                    print("Model weights loaded successfully.")
                
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("Successfully loaded optimizer state.")
                
                if 'max_fmap_score' in checkpoint:
                    max_fmap_score = checkpoint['max_fmap_score']
                    print(f"Restore the best score: {max_fmap_score}")
            else:
                print(f"Warning: Checkpoint file does not exist: {args.checkpoint}")
                print("Train from scratch")

        max_fmap_score, fmap_score = 0, -1
        for epoch in range(start_epoch, args.num_epochs):
            train_cross_view_epoch(epoch, train_dataloader, model, optimizer, criterion, writer, use_cuda, args)
            if epoch % args.validation_interval == 0:
                if validation_mode == 'multi_view':
                    if val_dataloader is not None:
                        fmap_score = val_cross_view_multi_view_epoch(cfg, epoch, val_dataloader, model, writer, use_cuda, args)
                        print(f'accuracy: {fmap_score:.4f}')
                    else:
                        print('Not created')
                        fmap_score = 0.0
                    
                elif validation_mode == 'single_view':
                    if single_view_dataloader is not None:
                        fmap_score = val_cross_view_multi_view_epoch(cfg, epoch, single_view_dataloader, model, writer, use_cuda, args)
                        print(f'auuuracy: {fmap_score:.4f}')
                    else:
                        print('Not created')
                        fmap_score = 0.0

                for f in os.listdir(save_dir):
                    file_path = os.path.join(save_dir, f)
                    if f.startswith("model_") and not f.startswith("BEST"):
                        os.remove(file_path)

                save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, fmap_score))
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)

                if max_fmap_score < fmap_score:
                    max_fmap_score = fmap_score
                    for f in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, f)
                        if f.startswith("BEST"):
                            os.remove(file_path)
                    save_file_path = os.path.join(save_dir, 'BESTmodel_{}_{:.4f}.pth'.format(epoch, fmap_score))
                    states = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(states, save_file_path)
            else:

                for f in os.listdir(save_dir):
                    file_path = os.path.join(save_dir, f)
                    if f.startswith("model_") and not f.startswith("BEST"):
                        os.remove(file_path)

                if fmap_score > 0: 
                    save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, fmap_score))
                else: 
                    save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))

                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
        return
    else:
        raise ValueError("Cross-view training mode (CS or CV) must be used.")