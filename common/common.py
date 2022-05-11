from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--dataset', help='Dataset', default='OWR',
                        choices=['OWR', 'COSDA-HR', 'CORe50'], type=str)
    parser.add_argument('--source', help="source dataset ROD/synROD", type=str, default="ROD")
    parser.add_argument('--target', help="target dataset ROD/synROD/arid", type=str, default="arid") # used only in eval
    parser.add_argument('--eval_episode', help="Episode to eval in incremental learning", type=int, default=0)
    parser.add_argument('--dataorder', choices=[0,1,2,3,4], type=int, default=0, help="Select classes order for incremental learning")
    parser.add_argument('--total_n_classes', help="Desired total number of classes that could be learned", default=50, type=int)

    parser.add_argument('--replay_buffer_size', default=2000, type=int, help="Size of replay buffer, 0 to disable")

    parser.add_argument('--resize_factor', type=float, default=0.08, help="Min resize factor for random resized crop")

    parser.add_argument('--load_path', help='Path to the loading checkpoint', default=None, type=str)
    parser.add_argument('--no_strict', help='Do not strictly load state_dicts', action='store_true')

    ##### Training Configurations #####
    parser.add_argument('--mode', help='training mode', choices=["Main_Trainer", "OWR_eval"], default="Main_Trainer")
    parser.add_argument('--optimizer', help='Optimizer', choices=['sgd', 'adam', 'lars'], default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler', choices=['step_decay', 'cosine'], default='step_decay', type=str)
    parser.add_argument('--warmup', help='Warm-up iterations', default=2500, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate', default=0.05, type=float)
    parser.add_argument('--weight_decay', help='Weight decay', default=1e-6, type=float)
    parser.add_argument('--num_workers', type=int, default=4, help="Num of workers for data loaders")


    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.07, type=float)
    parser.add_argument('--hyp_dims', type=int, help="Number of hypersphere dimensions", default=128, dest="simclr_dim")

    ##### Incremental learning options #####
    parser.add_argument('--compactness_margin', type = float, help="Margin for compactness/separation learning constraint", default=0.1) # epsilon in paper
    parser.add_argument('--ep_0_min_its', type=int, default=15000, help="Minimum number of iterations for first episode")
    parser.add_argument('--eps_min_its', type=int, default=15000, help="Minimum number of iterations for subsequent episodes")


    parser.add_argument("--suffix", default="", type=str, help="suffix for log dir")
    parser.add_argument("--save_dir", default="eval_output", type=str, help="Directory for eval outputs (features, prototypes)")


    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
