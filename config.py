import yaml
from types import SimpleNamespace

args = {
    "preprocess_config": "../vlsp2023-ess/config/preprocess.yaml",
    "train_config": "../vlsp2023-ess/config/train.yaml",
    "model_config": "../vlsp2023-ess/config/model.yaml",
}

args = SimpleNamespace(**args)

def read_config(args=args):
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    
    model_config["mode"] = "train"
    
    return preprocess_config, model_config, train_config


if __name__ == "__main__":
    print(read_config())