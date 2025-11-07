"""
Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/train.py
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Callable

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

import rich
from rich import pretty
from rich.traceback import install

pretty.install()
install()

# Allow local imports
@contextmanager
def chdir(path: str):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


_path = Path(__file__).parent.parent
sys.path.insert(0, str(_path))

with chdir(_path):
    import tria
    from tria.model.tria import TRIA
    from tria.pipelines.tokenizer import Tokenizer, TokenSequence
    from tria.features import rhythm_features
    from tria.model.mask import (
        get_span_mask, 
        get_current_codebook_mask, 
        get_next_codebooks_mask, 
        get_random_mask, 
        combine_masks,
        format_seed,
        cosine_schedule,
    )
    from tria.data.dataset import StemDataset
    from tria.util import count_parameters, print, exists, ensure_dir
    from tria import transforms
    from tria import constants

warnings.filterwarnings("ignore", category=UserWarning)

################################################################################
# Train TRIA for masked language modeling of acoustic tokens
################################################################################

# Ensure unique seeds per training batch, even if dataset "loops" back to start
IDX_OFFSET = 0

# Masked span proportions
MIN_SPAN_PROP = 0.
MAX_SPAN_PROP = 1.

# Feature dropout to enable CFG at inference
P_CFG_DROPOUT = 0.

# Enable cudnn autotuner to speed up training; uncomment to trade memory for speed
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))


# Optimizers
AdamW = argbind.bind(torch.optim.AdamW)
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind()
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Model
TRIA = argbind.bind(TRIA)

# Data
StemDataset = argbind.bind(StemDataset, "train", "val")

# Rhythm features
rhythm_features = argbind.bind(rhythm_features)

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    prob: float = 1.0,
    names: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    transform = transforms.Compose(*to_tfm(names), prob=prob)
    return transform


@dataclass
class State:
    model:      TRIA
    optimizer:  AdamW
    scheduler:  ExponentialLR
    tokenizer:  Tokenizer
    train_tfm:  transforms.Compose
    val_tfm:    transforms.Compose
    train_data: StemDataset
    val_data:   StemDataset
    tracker:    Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
):
    # Load/initialize model
    model, extras = TRIA(), {}
    tracker.print(model)
    print(f"Trainable parameters: {count_parameters(model)}")

    if resume:
        load_dir = f"{save_path}/{tag}"
        model_pth = Path(load_dir) / "model.pt"
        extras_pth = Path(load_dir) / "extras.pt"
        
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{load_dir}")
        if model_pth.exists():
            sd = torch.load(model_pth, map_location="cpu")
            model.load_state_dict(sd)
        
        if extras_pth.exists():
            extras = torch.load(extras_pth, map_location="cpu", weights_only=False)

    model = accel.prepare_model(model)

    # Load/initialize optimizer
    with argbind.scope(args):
        optimizer = AdamW(model.parameters(), use_zero=accel.use_ddp)
        scheduler = ExponentialLR(optimizer)

    if "optimizer" in extras:
        optimizer.load_state_dict(extras["optimizer"])
    if "scheduler" in extras:
        scheduler.load_state_dict(extras["scheduler"])
    if "tracker" in extras:
        tracker.load_state_dict(extras["tracker"])

    # Load datasets
    with argbind.scope(args, "train"):
        train_data = StemDataset()
    with argbind.scope(args, "val"):
        val_data = StemDataset()

    # Load data augmentations
    with argbind.scope(args, "train"):
        train_tfm = build_transform()
    with argbind.scope(args, "val"):
        val_tfm = build_transform()

    # Load tokenizer
    with argbind.scope(args):
        tokenizer = Tokenizer().to(accel.device)

    return State(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
        train_tfm=train_tfm,
        val_tfm=val_tfm,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):

    global MIN_SPAN_PROP
    global MAX_SPAN_PROP

    output = {}
    
    state.model.eval()

    batch = util.prepare_batch(batch, accel.device)

    signal = batch["drums"]["signal"]
    idx = batch["idx"]
    seed = format_seed(batch["idx"])

    # Tokenize signal
    tokens = state.tokenizer.encode(signal).tokens  # (n_batch, n_codebooks, n_frames)

    n_batch, n_codebooks, n_frames = tokens.shape

    # Apply data augmentation prior to rhythm feature extraction
    tfm_kwargs = state.val_tfm.batch_instantiate(idx.tolist(), signal.clone())
    signal_aug = state.val_tfm.transform(signal.clone(), **tfm_kwargs)

    # Extract rhythm features
    feats = rhythm_features(signal_aug)  # (n_batch, n_feats, n_frames')
    feats = torch.nn.functional.interpolate(
        feats, n_frames, mode=accel.unwrap(state.model).interp,
    )  # (n_batch, n_feats, n_frames)
    
    # Sample timestep, mask proportion, and codebooks
    t = torch.zeros(n_batch, device=accel.device, dtype=torch.float)
    for i, s in enumerate(seed):
        t[i] = torch.from_numpy(s.uniform(0, 1, (1,))).to(t)
    mp = cosine_schedule(t)
    cb = torch.zeros(n_batch, device=accel.device, dtype=torch.long)
    for i, s in enumerate(seed):
        cb[i] = torch.from_numpy(s.randint(0, n_codebooks, (1,))).to(cb)

    # DEBUG
    #cb[:] = cb[0]
        
    # Sample masks        
    span_mask = get_span_mask(tokens, MIN_SPAN_PROP, MAX_SPAN_PROP, idx)  # (n_batch, n_frames)
    current_codebook_mask = get_current_codebook_mask(tokens, cb)         # (n_batch, n_codebooks)
    next_codebooks_mask = get_next_codebooks_mask(tokens, cb)             # (n_batch, n_codebooks)
    rand_mask = get_random_mask(tokens, mp, idx)                          # (n_batch, n_codebooks, n_frames)

    feats_mask = ~span_mask  # (n_batch, n_frames)
    tokens_mask = combine_masks(
        span_mask, current_codebook_mask, next_codebooks_mask, rand_mask
    )  # (n_batch, n_codebooks, n_frames)

    loss_by_cb = {_cb: [] for _cb in cb.tolist()}
    acc_by_cb = {_cb: [] for _cb in cb.tolist()}
    
    with accel.autocast():
        logits = state.model(tokens, feats, cb, tokens_mask, feats_mask)

        # Compute loss only for masked token positions in current codebook
        loss_mask = torch.logical_and(current_codebook_mask[:, :, None], ~tokens_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 3, 1, 2), 
            tokens,
            reduction="none",
            label_smoothing=0.1,
        )  * loss_mask.float()  # (n_batch, n_codebooks, n_frames)
        
        output["loss/cross_entropy"] = loss.mean()

    # Log loss/accuracy by codebook
    acc = (
        (logits.argmax(dim=-1) == tokens).float() * loss_mask.float()
    ).sum(dim=(1,2)) / loss_mask.float().sum(dim=(1,2)).clamp_min(1.)  # (n_batch, n_codebooks)
                                                              
    for i, _cb in enumerate(cb.tolist()):
        loss_by_cb[_cb] += [loss[i].mean().item()]
        acc_by_cb[_cb] += [acc[i].mean().item()]

    def avg(d: dict):
        for k, v in d.items():
            d[k] = sum(v)/len(v)
        return d

    for k, v in avg(loss_by_cb).items():
        output[f"loss/codebook_{k}"] = v
    for k, v in avg(acc_by_cb).items():
        output[f"acc/codebook_{k}"] = v

    return output


@timer()
def train_loop(state, batch, accel):
    
    global IDX_OFFSET
    global MIN_SPAN_PROP
    global MAX_SPAN_PROP
    global P_CFG_DROPOUT
    
    state.model.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    signal = batch["drums"]["signal"]
    idx = batch["idx"] + IDX_OFFSET
    seed = format_seed(idx)
    
    with torch.no_grad():

        # Tokenize signal
        tokens = state.tokenizer.encode(signal).tokens  # (n_batch, n_codebooks, n_frames)
        n_batch, n_codebooks, n_frames = tokens.shape        

        # Apply data augmentation prior to rhythm feature extraction
        tfm_kwargs = state.train_tfm.batch_instantiate(idx.tolist(), signal.clone())
        signal_aug = state.train_tfm.transform(signal.clone(), **tfm_kwargs)

        # Extract rhythm features
        feats = rhythm_features(signal_aug)  # (n_batch, n_feats, n_frames')
        feats = torch.nn.functional.interpolate(
            feats, n_frames, mode=accel.unwrap(state.model).interp,
        )  # (n_batch, n_feats, n_frames)

    # Sample timestep, mask proportion, and codebooks
    t = torch.zeros(n_batch, device=accel.device, dtype=torch.float)
    for i, s in enumerate(seed):
        t[i] = torch.from_numpy(s.uniform(0, 1, (1,))).to(t)
    mp = cosine_schedule(t)
    cb = torch.zeros(n_batch, device=accel.device, dtype=torch.long)
    for i, s in enumerate(seed):
        cb[i] = torch.from_numpy(s.randint(0, n_codebooks, (1,))).to(cb)

    # DEBUG
    #cb[:] = cb[0]
        
    # Sample masks        
    span_mask = get_span_mask(tokens, MIN_SPAN_PROP, MAX_SPAN_PROP, idx)  # (n_batch, n_frames)
    current_codebook_mask = get_current_codebook_mask(tokens, cb)         # (n_batch, n_codebooks)
    next_codebooks_mask = get_next_codebooks_mask(tokens, cb)             # (n_batch, n_codebooks)
    rand_mask = get_random_mask(tokens, mp, idx)                          # (n_batch, n_codebooks, n_frames)

    feats_mask = ~span_mask  # (n_batch, n_frames)
    tokens_mask = combine_masks(
        span_mask, current_codebook_mask, next_codebooks_mask, rand_mask,
    )  # (n_batch, n_codebooks, n_frames)

    # Apply dropout to features to support CFG at inference
    drop_feats = torch.zeros(n_batch, device=accel.device, dtype=torch.bool)
    for i, s in enumerate(seed):
        drop_feats[i] = s.uniform(0, 1) > P_CFG_DROPOUT
    feats_mask = feats_mask * drop_feats[:, None]
    
    loss_by_cb = {_cb: [] for _cb in cb.tolist()}
    acc_by_cb = {_cb: [] for _cb in cb.tolist()}    
    
    with accel.autocast():
        logits = state.model(tokens, feats, cb, tokens_mask, feats_mask)  # (n_batch, n_codebooks, n_frames, codebook_size)

        # Compute loss only for masked token positions in current codebook
        loss_mask = torch.logical_and(current_codebook_mask[:, :, None], ~tokens_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 3, 1, 2),  # (n_batch, codebook_size, n_codebooks, n_frames)
            tokens,                      # (n_batch, n_codebooks, n_frames)
            reduction="none",
            label_smoothing=0.1,
        )  * loss_mask.float()           # (n_batch, n_codebooks, n_frames)
        
        output["loss/cross_entropy"] = loss.mean()
    
    # Log loss/accuracy by codebook
    with torch.no_grad():
        
        acc = (
            (logits.argmax(dim=-1) == tokens).float() * loss_mask.float()
        ).sum(dim=(1,2)) / loss_mask.float().sum(dim=(1,2)).clamp_min(1.)  # (n_batch, n_codebooks)
                                                                  
        for i, _cb in enumerate(cb.tolist()):
            loss_by_cb[_cb] += [loss[i].mean().item()]
            acc_by_cb[_cb] += [acc[i].mean().item()]

        def avg(d: dict):
            for k, v in d.items():
                d[k] = sum(v)/len(v)
            return d

        for k, v in avg(loss_by_cb).items():
            output[f"loss/codebook_{k}"] = v
        for k, v in avg(acc_by_cb).items():
            output[f"acc/codebook_{k}"] = v
    
    state.optimizer.zero_grad()
    accel.backward(output["loss/cross_entropy"])
    
    accel.scaler.unscale_(state.optimizer)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), 1e3
    )
    accel.step(state.optimizer)
    state.scheduler.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    # Update seed offset
    IDX_OFFSET += n_batch
    
    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    """
    Runs only on rank-0 process
    """
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "loss/cross_entropy"):
        state.tracker.print(f"Best model so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step}")

    for tag in tags:
        save_dir = f"{save_path}/{tag}"
        ensure_dir(save_dir)
        model_pth = Path(save_dir) / "model.pt"
        extras_pth = Path(save_dir) / "extras.pt"
        
        extras = {
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "tracker": state.tracker.state_dict(),
            "metadata": metadata,
        }
        torch.save(
            accel.unwrap(state.model).state_dict(),
            model_pth
        )
        torch.save(extras, extras_pth)


@torch.no_grad()
@argbind.bind(without_prefix=True)
def save_samples(
    state, 
    accel,
    sample_idx, 
    writer,
    # Inference params
    top_p: float = 0.85,
    top_k: int = None,
    temp: float = 1.0,
    mask_temp: float = 10.5,
    iterations: int = 8,
    guidance_scale: float = 2.0,
    causal_bias: float = 1.0,
):
    """
    Runs only on rank-0 process
    """

    global MIN_SPAN_PROP
    global MAX_SPAN_PROP
    
    state.tracker.print("Saving audio samples to TensorBoard")
    state.model.eval()

    # Obtain fixed samples
    samples = [state.val_data[idx] for idx in sample_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)

    signal = batch["drums"]["signal"]
    idx = batch["idx"]

    # Extract tokens
    _tokens = state.tokenizer.encode(signal)
    tokens = _tokens.tokens

    n_batch, n_codebooks, n_frames = tokens.shape    

    # Extract rhythm features
    feats = rhythm_features(signal)
    feats = torch.nn.functional.interpolate(
        feats, n_frames, mode=accel.unwrap(state.model).interp
    )
    
    # Construct (prefix) masks
    prefix_mask = torch.arange(
        n_frames, device=accel.device
    )[None, :].repeat(n_batch, 1) < int((1 - MAX_SPAN_PROP) * n_frames)
    tokens_mask = prefix_mask[:, None, :].repeat(1, n_codebooks, 1)
    feats_mask = ~prefix_mask

    # Batched inference
    generated = accel.unwrap(state.model).inference(
        tokens, feats, tokens_mask.clone(), feats_mask,
        top_p=top_p, top_k=top_k, temp=temp,
        mask_temp=mask_temp, iterations=iterations,
        guidance_scale=guidance_scale,
        causal_bias=causal_bias,
    )

    write_idx = ~tokens_mask
    tokens[write_idx] = generated[write_idx]

    # Decode tokens to audio
    _tokens.tokens = tokens
    out = state.tokenizer.decode(_tokens)
    
    audio_dict = {"generated": out}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer, "consolidate_state_dict"):
        state.optimizer.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "runs",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    val_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    sample_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    min_span_prop: float = 0.25,
    max_span_prop: float = 0.75,
    p_cfg_dropout: float = 0.2,
):

    global MIN_SPAN_PROP
    global MAX_SPAN_PROP
    MIN_SPAN_PROP, MAX_SPAN_PROP = min_span_prop, max_span_prop

    global P_CFG_DROPOUT
    P_CFG_DROPOUT = p_cfg_dropout
    
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path)
    
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
    )
    
    # Wrap functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            
            train_loop(state, batch, accel)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, accel, sample_idx, writer)

            if tracker.step % val_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
