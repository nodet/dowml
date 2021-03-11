from docplex.mp.model_reader import ModelReader


def run_model(path):
    print(f'Reading {path}...')
    m = ModelReader.read(path)
    # # Use compressed file on disk to store nodes
    # m.parameters.mip.strategy.file.set(3)
    # m.parameters.timelimit(10)
    # # Limit tree memory to 1 MB
    # m.parameters.workmem(1)
    m.solve(log_output=True)


if __name__ == '__main__':
    run_model('markshare1.mps.gz')
