from omegaconf import OmegaConf
from src.utils.parser import parse_args

if __name__ == "__main__":
    args = parse_args()

    # Load config file
    config = OmegaConf.load(args.config)

    if args.task == "train":
        from src.train import train

        train(config, args.config.split("/")[-1].split(".")[0]

    elif args.task == "test":
        from src.test import test

        eer = test(config, args.checkpoint)
        print("EER: {:.2f}% for {}.".format(eer * 100, args.checkpoint))

    else:
        raise NotImplementedError(f"{args.task} task not implemented")

    print("Done!")
