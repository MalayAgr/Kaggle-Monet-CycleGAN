import torch
from src import dataset, model, training
from src.config import Config, seed_everything
from torch.utils.data import DataLoader, Dataset


def init_dataloaders(
    train_data: Dataset, val_data: Dataset
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader


def main() -> None:
    transform = dataset.get_transforms(
        img_size=Config.IMG_SIZE,
        mean=Config.MEAN,
        std=Config.STD,
        training=True,
        additional_targets=Config.ADDITIONAL_TARGETS,
    )

    train_data, val_data = dataset.load_and_split_dataset(
        monet_dir=Config.filepath("monet"),
        photo_dir=Config.filepath("photo"),
        transform=transform,
        val_split=Config.VAL_SPLIT,
    )
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    train_loader, val_loader = init_dataloaders(
        train_data=train_data, val_data=val_data
    )

    cycle_gan = model.get_halved_model(device=Config.DEVICE)

    gen_opt, disc_opt = training.get_optimizers(cycle_gan=cycle_gan, lr=Config.LR)

    gen_sch, disc_sch = training.get_schedulers(gen_opt=gen_opt, disc_opt=disc_opt)

    trainer = training.Trainer(
        cycle_gan=cycle_gan,
        gen_opt=gen_opt,
        disc_opt=disc_opt,
        gen_sch=gen_sch,
        disc_sch=disc_sch,
        device=Config.DEVICE,
        use_amp=torch.cuda.is_available(),
    )

    cycle_gan, history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        scale=Config.LAMBDA,
    )


if __name__ == "__main__":
    seed_everything()
    main()
