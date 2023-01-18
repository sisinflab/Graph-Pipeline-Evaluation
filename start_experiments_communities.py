from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='allrecipes')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

allrecipes = ['user_0', 'user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_8', 'user_9',
              'user_10', 'user_11', 'user_12', 'user_13', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'item_5',
              'item_6', 'item_7', 'item_8', 'item_9']

bookcrossing = ['user_0', 'user_1', 'user_2', 'user_3', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'item_5']

if args.dataset == 'allrecipes':
    for data in allrecipes:
        print(f'Training on ALLRECIPES {data}...')
        run_experiment(f"config_files/experiment_communities_allrecipes.yml",
                       dataset='allrecipes_' + data,
                       gpu=args.gpu,
                       config_already_loaded=False)

elif args.dataset == 'bookcrossing':
    for data in allrecipes:
        print(f'Training on BOOKCROSSING {data}...')
        run_experiment(f"config_files/experiment_communities_bookcrossing.yml",
                       dataset='bookcrossing_' + data,
                       gpu=args.gpu,
                       config_already_loaded=False)

else:
    raise NotImplementedError('Dataset not available!')
