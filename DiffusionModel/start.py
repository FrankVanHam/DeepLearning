from pipeline import Pipeline

pl = Pipeline()
pl.load_data()
pl.load_pretraining()
#pl.train(1)
#pl.visualise_random_sample()
matrix = [
    # hero, non-hero, food, spell, side-facing
    [1, 0, 0,   0, 0],      
    [1, 0, 0.6, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0.1, 0, 0]
]
pl.visualise_context_sample(matrix)
