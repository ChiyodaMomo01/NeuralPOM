"""DDP full-state training with exact cursor/restart and two frozen stages."""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from neuralpom.v1.sampling import stage_samples
from neuralpom.v1.checkpoint import RestartStore, atomic_torch, cpu_tree, unwrap
from neuralpom.evaluation.protocol import atomic_json
from .runtime import AIRuntime
from .models import build_model
from .rollout import local_daily_update, forecast
from .state import standardized_loss
from .distributed import phase


def training_contract(runtime, stage):
    config = runtime.config
    return dict(runtime.contract(), stage=stage, training={key: config[key] for key in
        ("seed", "world_size", "stages", "optimizer")},
        execution={"deterministic_algorithms": True, "tf32": False},
        objective="equal standardized full-state channels, wet reference geometry",
        gradient_method="one_day_tbptt" if stage == "posttrain" else "direct_next_day")


def learning_rate(config, stage, update, total):
    cfg = config["optimizer"]
    if stage == "posttrain":
        return cfg["posttrain_lr"]*min(1., (update+1)/max(1, cfg["posttrain_warmup_updates"]))
    warm = config["optimizer"]["warmup_epochs"] * (total/config["stages"]["pretrain"]["max_epochs"])
    if update < warm:
        return cfg["pretrain_lr"]*(update+1)/max(1, warm)
    progress = min(1., (update-warm)/max(1., total-warm-1))
    return cfg["pretrain_final_lr"] + .5*(cfg["pretrain_lr"]-cfg["pretrain_final_lr"])*(1+math.cos(math.pi*progress))


def freeze(runtime, model, stage, step, selection, directory, filename=None):
    path = Path(directory)/"frozen"/(filename or f"{stage}.pt")
    atomic_torch(path, {"model": cpu_tree(unwrap(model).state_dict()), "stage": stage,
        "step": step, "selection": selection, "contract": runtime.contract(),
        "training_contract": training_contract(runtime, stage),
        "posttrain_gradient_mode": "one_day_tbptt" if stage == "posttrain" else None})
    return path


@torch.no_grad()
def validation(runtime, model, config, stage, rank, world, device):
    model.eval()
    network = unwrap(model)
    # BatchNorm running buffers must also represent the same frozen rank-0 model.
    if dist.is_initialized():
        for buffer in network.buffers():
            dist.broadcast(buffer, 0)
    samples = stage_samples(config, stage, "validation")
    days = 10 if stage == "posttrain" else 1
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for slot in range(0, len(samples), world):
        index = slot+rank
        def score():
            if index >= len(samples):
                return
            member, start = samples[index]
            initial = runtime.encoded(member, start)
            for day, prediction in forecast(runtime, network, initial, days):
                reference = runtime.data.load(member, start+day, device)
                target = runtime.normalizer.encode(runtime.codec.pack(reference))
                loss = standardized_loss(prediction, target, runtime.codec.weights(runtime.core, reference))
                totals[0].add_(loss.double()); totals[1].add_(1)
        phase(score, device)
    if dist.is_initialized():
        dist.all_reduce(totals)
    model.train()
    return float(totals[0]/totals[1])


def train(config, stage, rank, world, device, resume=False, max_updates=None):
    if stage not in ("pretrain", "posttrain"):
        raise ValueError("AI training has only two stages")
    runtime = phase(lambda: AIRuntime(config, device), device)
    model = phase(lambda: build_model(config["model"], runtime.codec).to(device), device)
    root = Path(config["run_root"])
    directory = root/stage
    parent = root/"frozen/pretrain.pt"
    def initialize():
        directory.mkdir(parents=True, exist_ok=True)
        if not resume and (directory/"resume.json").exists():
            raise FileExistsError("Existing AI training state requires --resume")
        if stage == "posttrain" and not resume:
            value = torch.load(parent, map_location="cpu", weights_only=False)
            if value["contract"] != runtime.contract() or value["stage"] != "pretrain":
                raise ValueError("AI Posttrain parent provenance differs")
            model.load_state_dict(value["model"], strict=True)
    phase(initialize, device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"][f"{stage}_lr"],
                                 weight_decay=config["optimizer"]["weight_decay"])
    contract = training_contract(runtime, stage)
    store = RestartStore(directory, contract)
    cursor = {"epoch": 0, "slot": 0, "lead": 0}
    step, best, stale, carried = 0, float("inf"), 0, None
    if resume:
        saved, physical = store.load(model, optimizer)
        cursor, step = saved["cursor"], saved["step"]
        best, stale = saved["extra"]["best"], saved["extra"]["stale"]
        carried = physical["encoded"].to(device) if physical is not None else None
    if resume and (directory/"COMPLETE.json").exists():
        complete = json.loads((directory/"COMPLETE.json").read_text())
        if complete["stage"] != stage or complete["step"] != step:
            raise ValueError("AI completion marker differs from committed restart")
        return {"status": "complete", "step": step}
    if max_updates is not None and step >= max_updates:
        return {"status": "paused_at_requested_update", "step": step}
    samples = stage_samples(config, stage)
    sampler = DistributedSampler(samples, num_replicas=world, rank=rank, seed=config["seed"], shuffle=True)
    settings = config["stages"][stage]
    epochs = settings.get("epochs", settings.get("max_epochs"))
    days = 10 if stage == "posttrain" else 1
    total = len(sampler)*epochs*days
    logger = None
    def tracking():
        nonlocal logger
        if rank == 0 and config["wandb"]["mode"] != "disabled":
            import wandb
            logger = wandb.init(project=config["wandb"]["project"], mode=config["wandb"]["mode"],
                entity=config["wandb"].get("entity"), job_type=stage,
                group=f"{config['core']}-{config['model']['architecture']}-seed{config['seed']}",
                dir=str(directory), name=f"{config['core']}-{config['model']['architecture']}-{stage}", config=config)
            atomic_json(directory/"tracking.json", {"mode": config["wandb"]["mode"],
                "id": logger.id, "url": logger.url if config["wandb"]["mode"] == "online" else None,
                "directory": logger.dir, "project": config["wandb"]["project"]})
        if rank == 0:
            atomic_json(directory/"config.json", config)
            atomic_json(directory/"contract.json", contract)
    phase(tracking, device)
    def checkpoint(next_cursor, physical=None):
        store.save(model, optimizer, next_cursor, step, physical=physical,
                   extra={"best": best, "stale": stale})
    try:
        for epoch in range(cursor["epoch"], epochs):
            if stale >= settings["patience"]:
                break
            model.train(); sampler.set_epoch(epoch)
            indices = list(sampler)
            first_slot = cursor["slot"] if epoch == cursor["epoch"] else 0
            for slot in range(first_slot, len(indices)):
                member, start = samples[indices[slot]]
                lead = cursor["lead"] if epoch == cursor["epoch"] and slot == first_slot else 0
                current = carried if lead else phase(lambda: runtime.encoded(member, start), device)
                for index in range(lead, days):
                    reference = phase(lambda: runtime.data.load(member, start+index+1, device), device)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate(config, stage, step, total)
                    # Input/target validation precedes DDP backward on every rank.
                    current, loss, norm = local_daily_update(runtime, model, current, reference, optimizer,
                                                              config["optimizer"]["clip_norm"])
                    logged_loss = loss.double().clone()
                    if dist.is_initialized():
                        dist.all_reduce(logged_loss)
                        logged_loss.div_(world)
                    step += 1
                    next_cursor = {"epoch": epoch, "slot": slot, "lead": index+1}
                    physical = {"encoded": current}
                    if index+1 == days:
                        next_cursor.update(slot=slot+1, lead=0); physical = None
                    if step % settings["checkpoint_every_updates"] == 0:
                        checkpoint(next_cursor, physical)
                    if rank == 0:
                        with (directory/"metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps({"step": step, "epoch": epoch, "loss": float(logged_loss), "rank0_loss": float(loss),
                                "grad_norm": float(norm), "learning_rate": optimizer.param_groups[0]["lr"]})+"\n")
                        if logger is not None:
                            logger.log({"train/loss": float(logged_loss), "train/grad_norm": float(norm),
                                "train/learning_rate": optimizer.param_groups[0]["lr"]}, step=step)
                    if max_updates is not None and step >= max_updates:
                        checkpoint(next_cursor, physical)
                        return {"status": "paused_at_requested_update", "step": step}
                carried = None
            val = validation(runtime, model, config, stage, rank, world, device)
            improved = val < best-settings.get("min_delta", 0.)
            best, stale = (val, 0) if improved else (best, stale+1)
            def save_epoch():
                if rank != 0:
                    return
                if stage == "pretrain" and improved:
                    freeze(runtime, model, stage, step, "validation_best", root)
                if stage == "posttrain" and improved:
                    freeze(runtime, model, stage, step, "validation_best_auxiliary", root, "posttrain_best.pt")
                epoch_record = {"epoch": epoch+1, "step": step, "validation_loss": val,
                    "best": best, "stale": stale}
                atomic_json(directory/"epoch.json", epoch_record)
                with (directory/"validation.jsonl").open("a") as stream:
                    stream.write(json.dumps(epoch_record)+chr(10))
                if logger is not None:
                    logger.log({"validation/loss": val}, step=step)
            phase(save_epoch, device)
            cursor = {"epoch": epoch+1, "slot": 0, "lead": 0}
            checkpoint(cursor)
            if stale >= settings["patience"]:
                break
        def finish():
            if rank != 0:
                return
            if stage == "posttrain":
                freeze(runtime, model, stage, step, "last_completed", root)
            atomic_json(directory/"COMPLETE.json", {"stage": stage, "step": step, "sampling": "hybrid_daily",
                "gradient_days": 1, "rollout_days": days, "physical_forward_steps": 0,
                "epochs_completed": cursor["epoch"], "best_validation_loss": best, "stale_epochs": stale,
                "termination": "validation_plateau" if stale >= settings["patience"] else "epoch_budget",
                "canonical_selection": "validation_best" if stage == "pretrain" else "last_completed",
                "auxiliary_best": "frozen/posttrain_best.pt" if stage == "posttrain" else None})
        phase(finish, device)
        return {"status": "complete", "step": step}
    finally:
        if logger is not None:
            logger.finish()
