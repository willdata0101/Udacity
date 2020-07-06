import argparse

def get_input_args():

    parser = argparse.ArgumentParser()

    # Train Args
    parser.add_argument('--data_dir', type=str, default='flowers', help='path to folder of flower images')
    parser.add_argument('--save_dir', action='store', default='save_directory', help='sets directory to save checkpoints')
    parser.add_argument('--arch', action='store', default='vgg19', help='choose architecture')
    parser.add_argument('--learning_rate', action='store', type=float, default='0.01', help='sets learning rate')
    parser.add_argument('--hidden_units', action='store', type=int, default='512', help='sets hidden units')
    parser.add_argument('--epochs', action='store', type=int, default='20', help='sets epochs')
    parser.add_argument('--gpu', action='store_true', default='gpu', help='use GPU for training')

    in_args = parser.parse_args()

    print("Argument 1: ", in_args.data_dir)
    print("Argument 2: ", in_args.save_dir)
    print("Argument 3: ", in_args.arch)
    print("Argument 4: ", in_args.learning_rate)
    print("Argument 5: ", in_args.hidden_units)
    print("Argument 6: ", in_args.epochs)
    print("Argument 7: ", in_args.gpu)

    # Predict Args
    parser.add_argument('--checkpoint', action='store', type=str, dest='predict.py', help='checkpoint filename')
    parser.add_argument('--topk', action='store', type=str, default='20', help='return top K most likely classes')
    parser.add_argument('--category_names', action='store', type=str, dest='cat_to_name.json', help='use mapping of categories to real names')
    #parser.add_argument('--gpu', action='store_true', default='gpu', help='use GPU for inference')
    
    return parser.parse_args()
