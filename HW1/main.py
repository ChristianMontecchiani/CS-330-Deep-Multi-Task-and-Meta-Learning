import argparse

import os
import importlib.util
import random
import torch
import numpy as np
import torch.nn.functional as F
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Check if submission module is present.  If it is not, then main() will not be executed.
use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
  from submission import DataGenerator, MANN


def meta_train_step(images, labels, model, optim, eval=False):

    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()


def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    if config.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif config.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: ", device)

    torch.manual_seed(config.random_seed)

    writer = SummaryWriter(f"runs/{config.num_classes}_{config.num_shot}_{config.random_seed}_{config.hidden_dim}")

    # Download Omniglot Dataset
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )
    assert os.path.isdir("./omniglot_resized")

    # Create Data Generator
    # This will sample meta-training and meta-testing tasks

    meta_train_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="train",
        cache=config.image_caching,
    )
    meta_train_loader = iter(
        torch.utils.data.DataLoader(
            meta_train_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    meta_test_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="test",
        cache=config.image_caching,
    )
    meta_test_loader = iter(
        torch.utils.data.DataLoader(
            meta_test_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    # Create model
    model = MANN(config.num_classes, config.num_shot + 1, config.hidden_dim)

    if(config.compile == True):
        try:
            model = torch.compile(model, backend=config.backend)
            print(f"MANN model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    import time

    times = []
    
    for step in tqdm(range(config.meta_train_steps)):
        ## Sample Batch
        ## Sample some meta-training tasks
        t0 = time.time()
        i, l = next(meta_train_loader)
        i, l = i.to(device), l.to(device)
        t1 = time.time()

        ## Train
        _, ls = meta_train_step(i, l, model, optim)
        t2 = time.time()
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])

        ## Evaluate
        ## Get meta-testing tasks
        if (step + 1) % config.eval_freq == 0:
            if config.debug == True:
                print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, l = next(meta_test_loader)
            i, l = i.to(device), l.to(device)
            pred, tls = meta_train_step(i, l, model, optim, eval=True)
            if config.debug == True:
                print("Train Loss:", ls.cpu().numpy(), "Test Loss:", tls.cpu().numpy())
            writer.add_scalar("Loss/test", tls, step)
            pred = torch.reshape(
                pred, [-1, config.num_shot + 1, config.num_classes, config.num_classes]
            )

            with open(f'submission/mann_results_{config.num_shot}_{config.num_classes}.npy', 'wb') as f:
                np.save(f, l.cpu().numpy())
                np.save(f, pred.cpu().numpy())

            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            l = torch.argmax(l[:, -1, :, :], axis=2)
            acc = pred.eq(l).sum().item() / (config.meta_batch_size * config.num_classes)
            if config.debug == True:
                print("Test Accuracy", acc)
            writer.add_scalar("Accuracy/test", acc, step)

            times = np.array(times)
            if config.debug == True:
                print(f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}")
            times = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_shot", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--meta_train_steps", type=int, default=25000)
    parser.add_argument("--image_caching", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cache', action='store_true')

    args = parser.parse_args()

    if args.cache == True:

        # Download Omniglot Dataset
        if not os.path.isdir("./omniglot_resized"):
            gdd.download_file_from_google_drive(
                file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
                dest_path="./omniglot_resized.zip",
                unzip=True,
            )
        assert os.path.isdir("./omniglot_resized")
    else:

        main(args)
