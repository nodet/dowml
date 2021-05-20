import os

from docplex.mp.model_reader import ModelReader


def is_a_cplex_model(f):
    l = f.lower()
    return (l.endswith('.mps') or
            l.endswith('.mps.gz') or
            l.endswith('.lp') or
            l.endswith('.lp.gz') or
            l.endswith('.sav') or
            l.endswith('.sav.gz'))


def is_a_parameter_file(f):
    l = f.lower()
    return l.endswith('.prm')


def run_model():
    model = None
    prm = None
    files = [f for f in os.listdir() if os.path.isfile(f)]
    for f in files:
        if not model and is_a_cplex_model(f):
            print(f'Found CPLEX model: {f}')
            model = f
        elif not prm and is_a_parameter_file(f):
            print(f'Found parameter file: {f}')
            prm = f
        else:
            print(f'Ignoring {f}')
    if not model:
        print(f'ERROR: no model found.')
        return
    m = ModelReader.read(model)
    params = None
    if prm:
        params = ModelReader.read_prm(prm)
    m.solve(log_output=True, cplex_parameters=params)


if __name__ == '__main__':
    run_model()
