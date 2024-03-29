from omegaconf import OmegaConf
from src.utils.parser import parse_args

if __name__ == "__main__":
    args = parse_args()

    # Load config file
    config = OmegaConf.load(args.config)

    if args.task == "train":
        from src.train import train

        train(config, args.config.split("/")[-1].split(".")[0], vocal=args.vocal)

    elif args.task == "test":
        from src.test import test

        test(config, args.checkpoint, args.test)

    else:
        raise NotImplementedError(f"{args.task} task not implemented")

    print("Done!")
