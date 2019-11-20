import os
import sys
import numpy as np
import tensorflow as tf

def train(env, model):
    """
    This function should train our model.
    """

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"MODEL_ONE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [MODEL_ONE]")
        exit()

    # Initialize model
    if sys.argv[1] == "MODEL_ONE":
        model = Reinforce(state_size, num_actions)
    elif:
        raise ValueError('Incorrect model arg')



if __name__ == '__main__':
    main()
