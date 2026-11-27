"""Train DVLT geometry on DSEC pairs, with optional F³ and REALM masking.

Base loss and shared K follow Déjà-Events III-B. The optional REALM arm
transfers the masking schedule, keeping the DVLT objective and special tokens.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distillation import (deja_events_distance, realm_activity_mask,
                          realm_patch_drop_probability, sample_shared_steps,
                          teacher_final_tokens)
from dsec_data import DSECTemporalPairs
from event_dvlt import EventDVLT, load_rgb, voxelize
from event_field import EventField


def main(args):
    if args.cover_train and not args.max_steps:
        raise ValueError('--cover-train requires --max-steps')
    if args.realm_progress_epochs is not None and (not args.realm or args.realm_progress_epochs < 1):
        raise ValueError('--realm-progress-epochs requires --realm and a positive value')
    if set(args.train) & set(args.val):
        raise ValueError('Training and validation sequences overlap')
    last_epoch = args.stop_after or args.epochs
    if not 1 <= last_epoch <= args.epochs:
        raise ValueError('--stop-after must be between 1 and --epochs')
    torch.manual_seed(args.seed)
    data = {seq: DSECTemporalPairs(args.root, seq) for seq in args.train + args.val}
    train = [(seq, pair) for seq in args.train for pair in data[seq].pairs]
    val = [(seq, pair) for seq in args.val for pair in data[seq].pairs]
    if not 2 <= args.max_views <= 18:
        raise ValueError('--max-views must be in [2,18]')
    clips = {seq: {start: tuple(range(start,start+2*views,2))
                   for start,_ in data[seq].pairs
                   for views in range(2,args.max_views+1)
                   if all(index in data[seq].ends for index in range(start,start+2*views,2))}
             for seq in args.train}
    if not train or not val:
        raise ValueError('Both splits need usable frame pairs')
    if args.val_limit and len(val) > args.val_limit:
        val = [val[i] for i in np.linspace(0, len(val)-1, args.val_limit, dtype=int)]
    if args.steps_per_epoch:
        train = train[:args.steps_per_epoch]
    horizon = args.max_steps or args.epochs*len(train)
    if args.cover_train:
        horizon = max(horizon,len(train))
    if horizon < 1:
        raise ValueError('Training horizon must be positive')
    if args.max_steps is not None:
        if args.stop_after:
            raise ValueError('--max-steps and --stop-after are mutually exclusive')
        last_epoch = math.ceil(horizon/len(train))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != 'resume'}
    config.update(train_pairs=len(train), val_pairs=len(val), schedule_steps=horizon, actual_epochs=last_epoch)
    if args.resume:
        recorded = json.loads((args.out/'config.json').read_text())
        recorded.setdefault('step_gating', 'saag')
        if any(recorded.get(k) != v for k,v in config.items() if k not in ('stop_after','actual_epochs')):
            raise ValueError('Resume requires the same data, model and LR schedule')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(config, flush=True)
    teacher = load_rgb(args.assets)
    student = EventDVLT(args.assets).cuda().eval()
    field_encoder = EventField(args.f3, args.assets/'f3.pth', 4) if args.f3 else None
    fast, slow = [], []
    adapter_names = []
    for name, p in student.named_parameters():
        if name.startswith(('gs_head.', 'geometry_features.', 'field_features.')) or name == 'log_metric_scale':
            p.requires_grad_(False)
        elif not field_encoder and name.startswith('event_adapter.'):
            p.requires_grad_(False)
        if p.requires_grad:
            adapter_names.append(name)
            if ((args.step_gating != 'saag' and name.startswith('corrections.')) or
                (args.step_gating == 'frozen' and '.depth_scale.proj.' in name)):
                p.requires_grad_(False)
            else:
                (slow if name.startswith('backbone.') else fast).append(p)
    opt = torch.optim.AdamW([{'params': slow, 'lr': 3e-5},
                             {'params': fast, 'lr': 3e-4}], weight_decay=.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, horizon)

    mask_rng = torch.Generator(device="cuda").manual_seed(args.seed+2) if args.max_steps else None

    @lru_cache(maxsize=128)
    def frame(seq, index):
        packet, rgb = data[seq].frame(index)
        voxel = voxelize(packet).half()
        field = (field_encoder(packet).half().cpu() if field_encoder else
                 torch.zeros((32,96,128), dtype=torch.float16))
        return voxel, field, torch.from_numpy(rgb)

    def inputs(item):
        seq, pair = item
        voxel, field, rgb = (torch.stack(x)[None].cuda() for x in zip(*(frame(seq, i) for i in pair)))
        return voxel.float(), field.float(), rgb.permute(0,1,4,2,3).float()/255

    def compare(voxel, field, target, steps, epoch=None):
        mask, input_mask = None, None
        if args.realm and epoch is not None:
            # Max-pool occupancy directly to DVLT's 12x16 grid, preserving isolated events.
            occupancy = F.adaptive_max_pool2d(voxel[0].ne(0).any(1, keepdim=True).float(), (12,16))
            spatial = realm_activity_mask(occupancy, epoch, patch_size=1)
            mask = F.pad(spatial, (student.backbone.patch_start_idx,0), value=True)
            drop = realm_patch_drop_probability(epoch)
            if drop:
                input_mask = torch.rand(spatial.shape, device=voxel.device, generator=mask_rng) < drop
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens, _, _ = student.geometry(voxel, field, steps=steps, input_mask=input_mask)
        return deja_events_distance(tokens, target, mask)

    @torch.no_grad()
    def evaluate(epoch):
        rows = []
        for seq, pair in val:
            voxel, field, rgb = inputs((seq, pair))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                target = teacher_final_tokens(teacher, rgb, 16)
            matched = compare(voxel, field, target, 16)
            empty = compare(torch.zeros_like(voxel), torch.zeros_like(field), target, 16)
            rows.append({'seq': seq, 'frames': pair, 'matched': float(matched), 'empty': float(empty)})
        report = args.out/f'val_{epoch:03}.json'
        temporary = report.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        temporary.replace(report)
        print('validation', epoch, {k: sum(r[k] for r in rows)/len(rows) for k in ('matched','empty')}, flush=True)

    step, first_epoch = 0, 1
    order_rng = torch.Generator().manual_seed(args.seed) if args.max_steps else None
    k_rng = torch.Generator().manual_seed(args.seed+1) if args.max_steps else None
    view_rng = torch.Generator().manual_seed(args.seed+3)
    params = dict(student.named_parameters())

    def save_checkpoint(path, epoch, training=None):
        state = {'model':{n:params[n].detach().cpu() for n in adapter_names},
                 'epoch':epoch,'step':step,'config':config}
        if training is not None:
            state['training'] = training
        tmp = path.with_suffix('.pt.tmp')
        torch.save(state,tmp)
        tmp.replace(path)

    def training_state(order=None, next_index=0):
        state = dict(optimizer=opt.state_dict(),schedule=schedule.state_dict(),
                     cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                     order_indices=order,next_index=next_index)
        state.update({name:rng.get_state() for name,rng in
                      [('order',order_rng),('views',view_rng),('loops',k_rng),('mask',mask_rng)]
                      if rng is not None})
        return state

    resume_order, resume_index = None, 0
    resume_path = args.out/'latest.pt'
    if args.resume:
        saved = torch.load(resume_path,map_location='cpu',weights_only=True)
        if 'training' not in saved:
            raise ValueError('This checkpoint predates optimizer and RNG checkpointing')
        if set(saved['model']) != set(adapter_names):
            raise ValueError('Adapter set differs from checkpoint')
        student.load_state_dict(saved['model'],strict=False)
        state = saved['training']
        opt.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['schedule'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        for name,rng in [('order',order_rng),('views',view_rng),('loops',k_rng),('mask',mask_rng)]:
            if rng is not None:
                rng.set_state(state[name])
        resume_order = state.get('order_indices')
        resume_index = state.get('next_index',0)
        step = saved['step']
        first_epoch = saved['epoch'] + (resume_order is None)
        log_path = args.out/'train.jsonl'
        retained = []
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row['step'] <= step:
                retained.append(line)
        if len(retained)!=step:
            raise ValueError('Training log does not cover the resume checkpoint')
        temporary = log_path.with_suffix('.resume.tmp')
        temporary.write_text('\n'.join(retained)+'\n')
        temporary.replace(log_path)
        if first_epoch > last_epoch:
            if not (args.out/f'val_{saved["epoch"]:03}.json').is_file():
                evaluate(saved['epoch'])
            return
    else:
        evaluate(0)
    for epoch in range(first_epoch, last_epoch+1):
        if resume_order is not None:
            order,start_index = resume_order,resume_index
            resume_order = None
        else:
            order,start_index = torch.randperm(len(train),generator=order_rng).tolist(),0
        for position in range(start_index,len(order)):
            index = order[position]
            if step >= horizon:
                break
            seq,pair = train[index]
            maximum = len(clips[seq][pair[0]])
            count = int(torch.randint(2,maximum+1,(1,),generator=view_rng))
            frames = clips[seq][pair[0]][:count]
            voxel, field, rgb = inputs((seq,frames))
            steps = sample_shared_steps(teacher,k_rng)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                target = teacher_final_tokens(teacher, rgb, steps)
            curriculum_epoch = epoch
            if args.realm_progress_epochs:
                curriculum_epoch = 1+(step*args.realm_progress_epochs)//horizon
            loss = compare(voxel, field, target, steps, curriculum_epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(slow+fast, 1., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            step += 1
            with (args.out/'train.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch, step=step, K=steps, loss=float(loss.detach()),
                                        grad_norm=float(norm), sequence=seq, frames=frames, curriculum_epoch=curriculum_epoch), allow_nan=False)+'\n')
            if step == 1 or step % 20 == 0:
                print('step', step, 'epoch', epoch, 'K', steps, 'loss', float(loss.detach()), flush=True)
            if args.snapshot_every and step % args.snapshot_every == 0:
                save_checkpoint(args.out/f'step_{step:06}.pt',epoch)
                save_checkpoint(resume_path,epoch,training_state(order,position+1))
        save_checkpoint(resume_path,epoch,training_state())
        if epoch == last_epoch or epoch % args.eval_every == 0:
            evaluate(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'assets', 'out'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--train', nargs='+', required=True)
    parser.add_argument('--val', nargs='+', required=True)
    parser.add_argument('--f3', type=Path)
    parser.add_argument('--realm', action='store_true')
    parser.add_argument('--step-gating', choices=('frozen','unfrozen','saag'), default='saag',
                        help='Freeze DVLT gates, train DVLT gates alone, or also train patch corrections')
    parser.add_argument('--max-steps', type=int, help='Exact step budget with an equally long cosine schedule')
    parser.add_argument('--cover-train',action='store_true',help='Raise the step budget if needed to visit every clip start')
    parser.add_argument('--realm-progress-epochs',type=int,help='Map REALM curriculum epochs onto optimizer progress')
    parser.add_argument('--snapshot-every',type=int,default=0,help='Save intermediate inference weights every N steps')
    parser.add_argument('--max-views', type=int, default=2, help='Sample temporal clips up to this many views')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--resume',action='store_true',help='Resume a saved step or epoch with optimizer and RNG state')
    parser.add_argument('--stop-after', type=int, help='Stop early while retaining the full cosine LR horizon')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--steps-per-epoch', type=int, default=0, help='Short integration runs')
    main(parser.parse_args())
