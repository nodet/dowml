import docplex.util.environment as environment
from docplex.mp.model_reader import ModelReader


def run_model(path):
    print(f'Reading {path}...')
    m = ModelReader.read(path)
    # Use compressed file on disk to store nodes
    m.parameters.mip.strategy.file.set(3)
    m.parameters.timelimit(10)
    # Limit tree memory to 1 MB
    m.parameters.workmem(1)

    # Dump the model as a file named local_model.lp
    m.export_as_lp("local_model.lp")
    # And add this as a job attachment with id model.lp
    environment.set_output_attachment('model.lp', 'local_model.lp')

    m.solve(log_output=True)

if __name__ == '__main__':
    run_model('markshare1.mps.gz')
