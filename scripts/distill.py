"""
Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/train.py

Distill TRIA masked language model:
- Freeze a loaded teacher TRIA model
- Train a (typically smaller) student TRIA model to match teacher logits
"""
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import argbind
import rich
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from rich import pretty
from rich.traceback import install
from torch.utils.tensorboard import SummaryWriter

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
    from tria import constants
    from tria import transforms


warnings.filterwarnings("ignore", category=UserWarning)

################################################################################
# Distill TRIA masked language model (teacher -> student)
################################################################################

# Ensure unique seeds per training batch, even if dataset "loops" back to start
IDX_OFFSET = 0

# Masked span proportions
MIN_SPAN_PROP = 0.0
MAX_SPAN_PROP = 1.0

# Feature dropout to enable CFG at inference (applied during distillation too)
P_CFG_DROPOUT = 0.0

# Enable cudnn autotuner to speed up training; uncomment to trade memory for speed
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))


# Optimizers
AdamW = argbind.bind(torch.optim.AdamW)
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind()
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
TRIA = argbind.bind(TRIA, "teacher", "student")

# Data
StemDataset = argbind.bind(StemDataset, "train", "val")

# Rhythm features
rhythm_features = argbind.bind(rhythm_features)

# Transforms: allow specification of separate augmentations for rhythm and
# timbre prompts at both train and validation
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "NormalizedBaseTransform",
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(
    transforms,
    "train_rhythm",
    "train_timbre",
    "val_rhythm",
    "val_timbre",
    filter_fn=filter_fn,
)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train_rhythm", "train_timbre", "val_rhythm", "val_timbre")
def build_transform(
    prob: float = 1.0,
    names: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    transform = transforms.Compose(*to_tfm(names), prob=prob)
    return transform


def _freeze_(model: torch.nn.Module):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _distill_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    eps: float = 1e-8,
):
    """
    KL( softmax(t/T) || softmax(s/T) ), averaged over masked positions only.

    Shapes:
      logits: (n_batch, n_codebooks, n_frames, n_vocab)
      mask:   (n_batch, n_codebooks, n_frames) boolean
    """
    # Flatten masked positions for stable reduction
    # (n_unmasked, n_vocab)
    s = student_logits[mask]
    t = teacher_logits[mask]

    if s.numel() == 0:
        return student_logits.new_tensor(0.0)

    T = float(temperature)
    s_logp = torch.nn.functional.log_softmax(s / T, dim=-1)
    t_p = torch.nn.functional.softmax(t / T, dim=-1)

    # KL(t || s) = sum t * (log t - log s)
    # torch.kl_div expects input: log-probs, target: probs
    kl = torch.nn.functional.kl_div(s_logp, t_p, reduction="batchmean")

    # Standard distillation scaling
    return kl * (T * T)


@dataclass
class State:
    teacher: TRIA
    student: TRIA
    optimizer: AdamW
    scheduler: ExponentialLR
    tokenizer: Tokenizer
    train_rhythm_tfm: transforms.Compose
    train_timbre_tfm: transforms.Compose
    val_rhythm_tfm: transforms.Compose
    val_timbre_tfm: transforms.Compose
    train_data: StemDataset
    val_data: StemDataset
    tracker: Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    # Teacher checkpoint
    teacher_path: str = None,
    teacher_tag: str = "best",
):
    # Load teacher (frozen)
    if teacher_path is None:
        raise ValueError(
            "Must provide --teacher_path=<runs_dir> containing <teacher_tag>/model.pt"
        )

    with argbind.scope(args, "teacher"):
        teacher, teacher_extras = TRIA(), {}
    tracker.print(teacher)
    print(f"Teacher parameters (total): {count_parameters(teacher)}")

    teacher_load_dir = f"{teacher_path}/{teacher_tag}"
    teacher_model_pth = Path(teacher_load_dir) / "model.pt"

    tracker.print(
        f"Loading teacher from {str(Path('.').absolute())}/{teacher_load_dir}"
    )
    if not teacher_model_pth.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_model_pth}")

    teacher_sd = torch.load(teacher_model_pth, map_location="cpu")
    teacher.load_state_dict(teacher_sd)
    teacher = teacher.to(accel.device)
    teacher = _freeze_(teacher)

    # Load / initialize student
    with argbind.scope(args, "student"):
        student, extras = TRIA(), {}
    tracker.print(student)
    print(f"Student trainable parameters: {count_parameters(student)}")

    if resume:
        load_dir = f"{save_path}/{tag}"
        model_pth = Path(load_dir) / "model.pt"
        extras_pth = Path(load_dir) / "extras.pt"

        tracker.print(f"Resuming student from {str(Path('.').absolute())}/{load_dir}")
        if model_pth.exists():
            sd = torch.load(model_pth, map_location="cpu")
            student.load_state_dict(sd)

        if extras_pth.exists():
            extras = torch.load(extras_pth, map_location="cpu", weights_only=False)

    student = accel.prepare_model(student)

    # Optimizer/scheduler (student only)
    with argbind.scope(args):
        optimizer = AdamW(student.parameters(), use_zero=accel.use_ddp)
        scheduler = ExponentialLR(optimizer)

    if "optimizer" in extras:
        optimizer.load_state_dict(extras["optimizer"])
    if "scheduler" in extras:
        scheduler.load_state_dict(extras["scheduler"])
    if "tracker" in extras:
        tracker.load_state_dict(extras["tracker"])

    # Data, transforms, tokenizer
    with argbind.scope(args, "train"):
        train_data = StemDataset()
    with argbind.scope(args, "val"):
        val_data = StemDataset()

    # Load data augmentations
    with argbind.scope(args, "train_rhythm"):
        train_rhythm_tfm = build_transform()
    with argbind.scope(args, "train_timbre"):
        train_timbre_tfm = build_transform()
    with argbind.scope(args, "val_rhythm"):
        val_rhythm_tfm = build_transform()
    with argbind.scope(args, "val_timbre"):
        val_timbre_tfm = build_transform()

    with argbind.scope(args):
        tokenizer = Tokenizer().to(accel.device)

    return State(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
        train_rhythm_tfm=train_rhythm_tfm,
        train_timbre_tfm=train_timbre_tfm,
        val_rhythm_tfm=val_rhythm_tfm,
        val_timbre_tfm=val_timbre_tfm,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel, distill_temp: float, hard_weight: float):
    global MIN_SPAN_PROP
    global MAX_SPAN_PROP

    output = {}

    state.teacher.eval()
    state.student.eval()

    batch = util.prepare_batch(batch, accel.device)

    signal = batch["drums"]["signal"]
    signal_lengths = batch["drums"]["signal_lengths"]

    idx = batch["idx"]
    seed = format_seed(batch["idx"])

    # Apply timbre prompt / target augmentation prior to tokenization
    timbre_prompt = signal.clone()
    timbre_tfm_kwargs = state.val_timbre_tfm.batch_instantiate(
        idx.tolist(), timbre_prompt
    )
    timbre_prompt = state.val_timbre_tfm.transform(timbre_prompt, **timbre_tfm_kwargs)

    # Tokenize timbre prompt / target
    tokens = state.tokenizer.encode(
        timbre_prompt
    ).tokens  # (n_batch, n_codebooks, n_frames)
    n_batch, n_codebooks, n_frames = tokens.shape

    tokens_lengths = (
        n_frames * signal_lengths.clone().float() / timbre_prompt.signal_length
    ).long()

    # Apply data augmentation prior to rhythm feature extraction
    rhythm_prompt = signal.clone()
    rhythm_tfm_kwargs = state.val_rhythm_tfm.batch_instantiate(
        idx.tolist(), rhythm_prompt
    )
    rhythm_prompt = state.val_rhythm_tfm.transform(rhythm_prompt, **rhythm_tfm_kwargs)

    # Extract rhythm features
    feats = rhythm_features(rhythm_prompt)  # (n_batch, n_feats, n_frames')
    feats = torch.nn.functional.interpolate(
        feats,
        n_frames,
        mode=accel.unwrap(state.model).interp,
    )  # (n_batch, n_feats, n_frames)

    # Sample timestep, mask proportion, and codebooks
    t = torch.zeros(n_batch, device=accel.device, dtype=torch.float)
    for i, s in enumerate(seed):
        t[i] = torch.from_numpy(s.uniform(0, 1, (1,))).to(t)
    mp = cosine_schedule(t)
    cb = torch.zeros(n_batch, device=accel.device, dtype=torch.long)
    for i, s in enumerate(seed):
        cb[i] = torch.from_numpy(s.randint(0, n_codebooks, (1,))).to(cb)

    # Sample masks
    span_mask = get_span_mask(
        tokens, MIN_SPAN_PROP, MAX_SPAN_PROP, idx, tokens_lengths
    )  # (n_batch, n_frames)
    current_codebook_mask = get_current_codebook_mask(
        tokens, cb
    )  # (n_batch, n_codebooks)
    next_codebooks_mask = get_next_codebooks_mask(tokens, cb)  # (n_batch, n_codebooks)
    rand_mask = get_random_mask(tokens, mp, idx)  # (n_batch, n_codebooks, n_frames)

    feats_mask = ~span_mask  # (n_batch, n_frames)
    tokens_mask = combine_masks(
        span_mask, current_codebook_mask, next_codebooks_mask, rand_mask
    )  # (n_batch, n_codebooks, n_frames)

    # Compute loss only for masked token positions in current codebook
    lengths_mask = (
        torch.arange(n_frames, dtype=torch.long, device=accel.device)[None, None, :]
        < tokens_lengths[:, None, None]
    )
    loss_mask = torch.logical_and(current_codebook_mask[:, :, None], ~tokens_mask)
    loss_mask = torch.logical_and(loss_mask, lengths_mask)

    # Denominator for masked-normalized losses
    den = loss_mask.float().sum().clamp_min(1.0)

    with accel.autocast():
        # Teacher logits (frozen)
        teacher_logits = state.teacher(
            tokens, feats, cb, tokens_mask, feats_mask, lengths=tokens_lengths
        )  # (n_batch, n_codebooks, n_frames, n_vocab)

        # Student logits
        student_logits = state.student(
            tokens, feats, cb, tokens_mask, feats_mask, lengths=tokens_lengths
        )  # (n_batch, n_codebooks, n_frames, n_vocab)

        # Distillation KL (already averaged over masked positions inside helper)
        kl = _distill_kl_loss(
            student_logits, teacher_logits, loss_mask, temperature=distill_temp
        )
        output["loss/distill_kl"] = kl

        # Optional hard CE for monitoring (and optionally mixing), normalized over masked positions
        ce_per_pos = torch.nn.functional.cross_entropy(
            student_logits.permute(
                0, 3, 1, 2
            ),  # (n_batch, n_vocab, n_codebooks, n_frames)
            tokens,  # (n_batch, n_codebooks, n_frames)
            reduction="none",
            label_smoothing=0.0,
        )  # (n_batch, n_codebooks, n_frames)
        ce = (ce_per_pos * loss_mask.float()).sum() / den
        output["loss/hard_cross_entropy"] = ce

        output["loss/total"] = (1.0 - hard_weight) * kl + hard_weight * ce

    # Log accuracy by codebook (student)
    with torch.no_grad():
        acc = (
            (student_logits.argmax(dim=-1) == tokens).float() * loss_mask.float()
        ).sum(dim=(1, 2)) / loss_mask.float().sum(dim=(1, 2)).clamp_min(1.0)

        # Log CE/acc by selected codebook (cb) for interpretability
        loss_by_cb = {_cb: [] for _cb in cb.tolist()}
        acc_by_cb = {_cb: [] for _cb in cb.tolist()}

        # Per-sample masked-normalized CE (so it matches the global normalization style)
        den_i = loss_mask.float().sum(dim=(1, 2)).clamp_min(1.0)  # (n_batch,)
        ce_i = (ce_per_pos * loss_mask.float()).sum(dim=(1, 2)) / den_i  # (n_batch,)

        for i, _cb in enumerate(cb.tolist()):
            loss_by_cb[_cb] += [ce_i[i].item()]
            acc_by_cb[_cb] += [acc[i].mean().item()]

        def avg(d: dict):
            for k, v in d.items():
                d[k] = sum(v) / len(v)
            return d

        for k, v in avg(loss_by_cb).items():
            output[f"loss_ce/codebook_{k}"] = v
        for k, v in avg(acc_by_cb).items():
            output[f"acc/codebook_{k}"] = v

    return output


@timer()
def train_loop(state, batch, accel, distill_temp: float, hard_weight: float):
    global IDX_OFFSET
    global MIN_SPAN_PROP
    global MAX_SPAN_PROP
    global P_CFG_DROPOUT

    state.teacher.eval()
    state.student.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)

    signal = batch["drums"]["signal"]
    signal_lengths = batch["drums"]["signal_lengths"]

    idx = batch["idx"] + IDX_OFFSET
    seed = format_seed(idx)

    with torch.no_grad():
        # Apply timbre prompt / target augmentation prior to tokenization
        timbre_prompt = signal.clone()
        timbre_tfm_kwargs = state.train_timbre_tfm.batch_instantiate(
            idx.tolist(), timbre_prompt
        )
        timbre_prompt = state.train_timbre_tfm.transform(
            timbre_prompt, **timbre_tfm_kwargs
        )

        # Tokenize signal
        tokens = state.tokenizer.encode(
            timbre_prompt
        ).tokens  # (n_batch, n_codebooks, n_frames)
        n_batch, n_codebooks, n_frames = tokens.shape

        tokens_lengths = (
            n_frames * signal_lengths.clone().float() / timbre_prompt.signal_length
        ).long()

        # Apply data augmentation prior to rhythm feature extraction
        rhythm_prompt = signal.clone()
        rhythm_tfm_kwargs = state.train_rhythm_tfm.batch_instantiate(
            idx.tolist(), rhythm_prompt
        )
        rhythm_prompt = state.train_rhythm_tfm.transform(
            rhythm_prompt, **rhythm_tfm_kwargs
        )

        # Extract rhythm features
        feats = rhythm_features(rhythm_prompt)  # (n_batch, n_feats, n_frames')
        feats = torch.nn.functional.interpolate(
            feats,
            n_frames,
            mode=accel.unwrap(state.model).interp,
        )  # (n_batch, n_feats, n_frames)

    # Log padding (invalid) proportion of batch
    output.update(
        {f"memory/pad_amt": 1 - (tokens_lengths.sum() / (n_batch * n_frames))}
    )

    # Sample timestep, mask proportion, and codebooks
    t = torch.zeros(n_batch, device=accel.device, dtype=torch.float)
    for i, s in enumerate(seed):
        t[i] = torch.from_numpy(s.uniform(0, 1, (1,))).to(t)
    mp = cosine_schedule(t)
    cb = torch.zeros(n_batch, device=accel.device, dtype=torch.long)
    for i, s in enumerate(seed):
        cb[i] = torch.from_numpy(s.randint(0, n_codebooks, (1,))).to(cb)

    # Sample masks
    span_mask = get_span_mask(
        tokens, MIN_SPAN_PROP, MAX_SPAN_PROP, idx, tokens_lengths
    )  # (n_batch, n_frames)
    current_codebook_mask = get_current_codebook_mask(
        tokens, cb
    )  # (n_batch, n_codebooks)
    next_codebooks_mask = get_next_codebooks_mask(tokens, cb)  # (n_batch, n_codebooks)
    rand_mask = get_random_mask(tokens, mp, idx)  # (n_batch, n_codebooks, n_frames)

    feats_mask = ~span_mask  # (n_batch, n_frames)
    tokens_mask = combine_masks(
        span_mask,
        current_codebook_mask,
        next_codebooks_mask,
        rand_mask,
    )  # (n_batch, n_codebooks, n_frames)

    # Apply dropout to features to support CFG at inference
    drop_feats = torch.zeros(n_batch, device=accel.device, dtype=torch.bool)
    for i, s in enumerate(seed):
        drop_feats[i] = s.uniform(0, 1) > P_CFG_DROPOUT
    feats_mask = feats_mask * drop_feats[:, None]

    # Compute loss only for masked token positions in current codebook
    lengths_mask = (
        torch.arange(n_frames, dtype=torch.long, device=accel.device)[None, None, :]
        < tokens_lengths[:, None, None]
    )
    loss_mask = torch.logical_and(current_codebook_mask[:, :, None], ~tokens_mask)
    loss_mask = torch.logical_and(loss_mask, lengths_mask)

    # Denominator for masked-normalized losses
    den = loss_mask.float().sum().clamp_min(1.0)

    with accel.autocast():
        with torch.no_grad():
            teacher_logits = state.teacher(
                tokens, feats, cb, tokens_mask, feats_mask, lengths=tokens_lengths
            )  # (n_batch, n_codebooks, n_frames, n_vocab)

        student_logits = state.student(
            tokens, feats, cb, tokens_mask, feats_mask, lengths=tokens_lengths
        )  # (n_batch, n_codebooks, n_frames, n_vocab)

        # Distillation KL (already averaged over masked positions inside helper)
        kl = _distill_kl_loss(
            student_logits, teacher_logits, loss_mask, temperature=distill_temp
        )
        output["loss/distill_kl"] = kl

        # Optional hard CE (mix-in), normalized over masked positions
        ce_per_pos = torch.nn.functional.cross_entropy(
            student_logits.permute(
                0, 3, 1, 2
            ),  # (n_batch, n_vocab, n_codebooks, n_frames)
            tokens,  # (n_batch, n_codebooks, n_frames)
            reduction="none",
            label_smoothing=0.0,
        )  # (n_batch, n_codebooks, n_frames)
        ce = (ce_per_pos * loss_mask.float()).sum() / den
        output["loss/hard_cross_entropy"] = ce

        output["loss/total"] = (1.0 - hard_weight) * kl + hard_weight * ce

    # Log loss/accuracy by codebook (student)
    with torch.no_grad():
        acc = (
            (student_logits.argmax(dim=-1) == tokens).float() * loss_mask.float()
        ).sum(dim=(1, 2)) / loss_mask.float().sum(dim=(1, 2)).clamp_min(1.0)

        loss_by_cb = {_cb: [] for _cb in cb.tolist()}
        acc_by_cb = {_cb: [] for _cb in cb.tolist()}

        # Per-sample masked-normalized CE (so it matches the global normalization style)
        den_i = loss_mask.float().sum(dim=(1, 2)).clamp_min(1.0)  # (n_batch,)
        ce_i = (ce_per_pos * loss_mask.float()).sum(dim=(1, 2)) / den_i  # (n_batch,)

        for i, _cb in enumerate(cb.tolist()):
            loss_by_cb[_cb] += [ce_i[i].item()]
            acc_by_cb[_cb] += [acc[i].mean().item()]

        def avg(d: dict):
            for k, v in d.items():
                d[k] = sum(v) / len(v)
            return d

        for k, v in avg(loss_by_cb).items():
            output[f"loss_ce/codebook_{k}"] = v
        for k, v in avg(acc_by_cb).items():
            output[f"acc/codebook_{k}"] = v

    state.optimizer.zero_grad()
    accel.backward(output["loss/total"])

    accel.scaler.unscale_(state.optimizer)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.student.parameters(), 1e3
    )
    accel.step(state.optimizer)
    state.scheduler.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    # Update seed offset
    IDX_OFFSET += n_batch

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path, accel):
    """
    Runs only on rank-0 process
    """
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "loss/total"):
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
        torch.save(accel.unwrap(state.student).state_dict(), model_pth)
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

    Note: uses the *student* model for inference every time.
          Logs *teacher* generations only on the first logging iteration.
    """
    global MIN_SPAN_PROP
    global MAX_SPAN_PROP

    state.tracker.print("Saving audio samples to TensorBoard")
    state.student.eval()
    state.teacher.eval()

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

    # Extract rhythm features (use student interp for features used by student inference)
    feats = rhythm_features(signal)
    feats = torch.nn.functional.interpolate(
        feats, n_frames, mode=accel.unwrap(state.student).interp
    )

    # Construct (prefix) masks
    prefix_mask = torch.arange(n_frames, device=accel.device)[None, :].repeat(
        n_batch, 1
    ) < int((1 - MAX_SPAN_PROP) * n_frames)
    tokens_mask = prefix_mask[:, None, :].repeat(1, n_codebooks, 1)
    feats_mask = ~prefix_mask

    # Batched inference (student)
    student_tokens = tokens.clone()
    generated_student = accel.unwrap(state.student).inference(
        student_tokens,
        feats,
        tokens_mask.clone(),
        feats_mask,
        top_p=top_p,
        top_k=top_k,
        temp=temp,
        mask_temp=mask_temp,
        iterations=iterations,
        guidance_scale=guidance_scale,
        causal_bias=causal_bias,
    )

    write_idx = ~tokens_mask
    student_tokens[write_idx] = generated_student[write_idx]

    # Decode tokens to audio (student)
    _tokens.tokens = student_tokens
    out_student = state.tokenizer.decode(_tokens)

    audio_dict = {"student/generated": out_student}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

        # Batched inference (teacher) - only at step 0
        # If teacher has a different interp mode, match it for teacher features.
        feats_teacher = rhythm_features(signal)
        feats_teacher = torch.nn.functional.interpolate(
            feats_teacher, n_frames, mode=getattr(state.teacher, "interp", "nearest")
        )

        teacher_tokens = tokens.clone()
        generated_teacher = state.teacher.inference(
            teacher_tokens,
            feats_teacher,
            tokens_mask.clone(),
            feats_mask,
            top_p=top_p,
            top_k=top_k,
            temp=temp,
            mask_temp=mask_temp,
            iterations=iterations,
            guidance_scale=guidance_scale,
            causal_bias=causal_bias,
        )

        teacher_tokens[write_idx] = generated_teacher[write_idx]

        # Decode tokens to audio (teacher)
        _tokens.tokens = teacher_tokens
        out_teacher = state.tokenizer.decode(_tokens)

        audio_dict["teacher/generated"] = out_teacher

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )


def validate(state, val_dataloader, accel, distill_temp: float, hard_weight: float):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel, distill_temp, hard_weight)
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
    # Distillation params
    distill_temp: float = 2.0,
    hard_weight: float = 0.0,
):
    """
    distill_temp:
      Temperature used for KL distillation; typical values: 1-4.

    hard_weight:
      Mix-in weight for hard CE on true tokens.
      total_loss = (1 - hard_weight) * KL + hard_weight * CE
    """
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
            train_loop(state, batch, accel, distill_temp, hard_weight)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, accel, sample_idx, writer)

            if tracker.step % val_freq == 0 or last_iter:
                validate(state, val_dataloader, accel, distill_temp, hard_weight)
                checkpoint(state, save_iters, save_path, accel)
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
