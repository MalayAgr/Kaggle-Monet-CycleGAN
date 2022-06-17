import contextlib
from typing import Any, Callable

import torch
from torch.cuda import amp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import discriminator_loss, generator_loss
from .model import CycleGAN

BackwardFunctionType = Callable[[torch.Tensor, Optimizer, amp.GradScaler | None], None]


class History:
    def __init__(self, metrics: list[str], record_valid: bool = True) -> None:
        self.metrics = metrics

        self.history: dict[str, dict] = {}
        self.history["train"] = {metric: [] for metric in metrics}

        if record_valid is True:
            self.history["valid"] = {metric: [] for metric in metrics}

    def __repr__(self) -> str:
        return repr(self.history)

    def __str__(self) -> str:
        return str(self.history)

    def __getitem__(self, key: str) -> Any:
        return self.history[key]

    def __len__(self) -> int:
        return len(self.history)

    def update(self, value: Any, metric: str, dataset: str = "train") -> None:
        if dataset not in self.history:
            msg = (
                f"No history recorded for dataset {dataset!r}. "
                f"Available datasets: {set(self.history.keys())}"
            )
            raise KeyError(msg)

        if metric not in self.metrics:
            msg = (
                f"No history recorded for metric {metric!r}. "
                f"Available metrics: {self.metrics}."
            )
            raise KeyError(msg)

        self.history[dataset][metric].append(value)


class Trainer:
    def __init__(
        self,
        cycle_gan: CycleGAN,
        gen_opt: Optimizer,
        disc_opt: Optimizer,
        gen_sch: _LRScheduler = None,
        disc_sch: _LRScheduler = None,
        device: torch.device = None,
        use_amp: bool = True,
    ) -> None:
        if not torch.cuda.is_available() and use_amp is True:
            raise ValueError(
                "use_amp is set to True even though CUDA is not available."
            )

        self.cycle_gan = cycle_gan
        self.device = device

        self.gen_opt = gen_opt
        self.gen_sch = gen_sch

        self.disc_opt = disc_opt
        self.disc_sch = disc_sch

        self.use_amp = use_amp

        # The nullcontext is a context manager that does nothing
        self.cm = amp.autocast() if use_amp is True else contextlib.nullcontext()

        self.device = torch.device("cpu") if device is None else device

        self.gen_scaler, self.disc_scaler = (
            (amp.GradScaler(), amp.GradScaler())
            if self.use_amp is True
            else (None, None)
        )

        self.backward_fn = self._make_backward_fn()

    def _forward(
        self, img: torch.Tensor, monet: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.cycle_gan(img, monet)

        # Calculate total discriminator loss
        total_disc_loss = discriminator_loss(
            real_img_disc=outputs["real_img_disc"],
            fake_img_disc=outputs["fake_img_disc"].detach(),
            real_monet_disc=outputs["real_monet_disc"],
            fake_monet_disc=outputs["fake_monet_disc"].detach(),
        )

        # Calculate total generator loss
        total_gen_loss = generator_loss(
            img=img,
            fake_img_disc=outputs["fake_img_disc"].detach(),
            cycled_img=outputs["cycled_img"],
            identity_img=outputs["identity_img"],
            monet=monet,
            fake_monet_disc=outputs["fake_monet_disc"].detach(),
            cycled_monet=outputs["cycled_monet"],
            identity_monet=outputs["identity_monet"],
        )

        return total_disc_loss, total_gen_loss

    def _make_backward_fn(self) -> BackwardFunctionType:
        def no_amp(
            loss: torch.Tensor, opt: Optimizer, scaler: amp.GradScaler = None
        ) -> None:
            loss.backward()
            opt.step()

        def with_amp(
            loss: torch.Tensor, opt: Optimizer, scaler: amp.GradScaler = None
        ) -> None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        return with_amp if self.use_amp is True else no_amp

    def _train_one_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.cycle_gan.train()

        p_loader = tqdm(loader, unit="batch", desc="Training")

        average_disc = torch.tensor(0.0, device=self.device)
        average_gen = torch.tensor(0.0, device=self.device)

        for idx, batch in enumerate(p_loader):
            # Send inputs to appropriate device
            for k, img in batch.items():
                batch[k] = img.to(self.device)

            img = batch["image"]
            monet = batch["monet_img"]

            with self.cm:
                total_disc_loss, total_gen_loss = self._forward(img=img, monet=monet)

                average_disc += total_disc_loss

                average_gen += total_gen_loss

            self.backward_fn(
                loss=total_disc_loss, opt=self.disc_opt, scaler=self.disc_scaler
            )

            self.backward_fn(
                loss=total_gen_loss, opt=self.gen_opt, scaler=self.gen_scaler
            )

            p_loader.set_postfix(
                disc_loss=total_disc_loss.item(), gen_loss=total_gen_loss.item()
            )

        average_gen = (average_gen / (idx + 1)).item()
        average_disc = (average_disc / (idx + 1)).item()

        return average_gen, average_disc

    def _validate_one_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.cycle_gan.eval()

        p_loader = tqdm(loader, unit="batch", desc="Validating")

        average_disc = torch.tensor(0.0, device=self.device)
        average_gen = torch.tensor(0.0, device=self.device)

        for idx, batch in enumerate(p_loader):
            # Send inputs to appropriate device
            for k, img in batch.items():
                batch[k] = img.to(self.device)

            img = batch["image"]
            monet = batch["monet_img"]

            total_disc_loss, total_gen_loss = self._forward(img=img, monet=monet)

            average_disc += total_disc_loss

            average_gen += total_gen_loss

            p_loader.set_postfix(
                disc_loss=total_disc_loss.item(), gen_loss=total_gen_loss.item()
            )

        average_gen = (average_gen / (idx + 1)).item()
        average_disc = (average_disc / (idx + 1)).item()

        return average_gen, average_disc

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 200
    ) -> tuple[CycleGAN, History]:
        history = History(
            metrics=["gen_loss", "disc_loss"], record_valid=val_loader is not None
        )

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}:")

            # Train
            gen_loss, disc_loss = self._train_one_epoch(loader=train_loader)

            history.update(gen_loss, metric="gen_loss", dataset="train")
            history.update(disc_loss, metric="disc_loss", dataset="train")

            print(f"Avg gen. loss={gen_loss}; Avg disc. loss={disc_loss}\n")

            if val_loader is not None:
                # Turn off gradients and validate
                with torch.no_grad():
                    gen_loss, disc_loss = self._validate_one_epoch(loader=val_loader)

                history.update(gen_loss, metric="gen_loss", dataset="valid")
                history.update(disc_loss, metric="disc_loss", dataset="valid")

                print(f"Avg gen. loss={gen_loss}; Avg disc. loss={disc_loss}\n")

            if (gen_sch := self.gen_sch) is not None:
                gen_sch.step()

            if (disc_sch := self.disc_sch) is not None:
                disc_sch.step()

        return self.cycle_gan, history
