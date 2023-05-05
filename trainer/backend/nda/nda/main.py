from args import get_args
from trainer import trainer

if __name__ == "__main__":
    args = get_args()
    T = trainer(args)

    if not args.eval:
        T.train()
    else:
        T.infer()
