import yaml
from src.utils.parser import parse_args

if __name__ == "__main__":
    args = parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.task == "train":
        from src.train import train

        train(config)

    elif args.task == "test":
        from src.test import test

        test(config)

    else:
        raise NotImplementedError(f"{args.task} task not implemented")

    print("Done!")
