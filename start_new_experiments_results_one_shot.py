from elliot.run import run_experiment

run_experiment(f"config_files/one_shot_rp3beta.yml",
               dataset='amazon-book',
               gpu=0,
               config_already_loaded=False)
