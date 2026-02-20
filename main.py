import sys
from train_baseline import main

if __name__ == "__main__":
    eval_only = "--eval" in sys.argv
    main(eval_only=eval_only)