import sklearn.metrics as skmt

from mlrose_hiive import NNClassifier, relu
from mlrose_hiive.decorators import short_name
from mlrose_hiive.runners._nn_runner_base import _NNRunnerBase
from mlrose_hiive.decorators import get_short_name
import numpy as np
import inspect as lk

"""
Example usage:
    from mlrose_hiive.runners import NNGSRunner

    grid_search_parameters = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],                     # nn params
        'learning_rate': [0.001, 0.002, 0.003],                         # nn params
        'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
    })

    nnr = NNGSRunner(x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     experiment_name='nn_test',
                     algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=500,
                     generate_curves=True,
                     seed=200972)

    results = nnr.run()          # GridSearchCV instance returned    
"""


@short_name('nngs')
class NNGSRunner(_NNRunnerBase):

    def __init__(self, x_train, y_train, x_test, y_test, experiment_name, seed, iteration_list, algorithm,
                 grid_search_parameters, grid_search_scorer_method=skmt.balanced_accuracy_score,
                 bias=True, early_stopping=True, clip_max=1e+10, activation=None,
                 max_attempts=500, n_jobs=1, cv=5, generate_curves=True, output_directory=None, extra_callback=None, extra_callback_info=None,
                 **kwargs):

        # update short name based on algorithm
        self._set_dynamic_runner_name(f'{get_short_name(self)}_{get_short_name(algorithm)}')

        self.extra_callback = extra_callback
        self.extra_callback_info = extra_callback_info

        # take a copy of the grid search parameters
        grid_search_parameters = {**grid_search_parameters}

        # hack for compatibility purposes
        if 'max_iter' in grid_search_parameters:
            grid_search_parameters['max_iter'] = grid_search_parameters.pop('max_iters')

        # call base class init
        super().__init__(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         experiment_name=experiment_name,
                         seed=seed,
                         iteration_list=iteration_list,
                         grid_search_parameters=grid_search_parameters,
                         generate_curves=generate_curves,
                         output_directory=output_directory,
                         n_jobs=n_jobs,
                         cv=cv,
                         grid_search_scorer_method=grid_search_scorer_method,
                         **kwargs)

        # build the classifier
        self.classifier = NNClassifier(runner=self,
                                       algorithm=algorithm,
                                       max_attempts=max_attempts,
                                       clip_max=clip_max,
                                       early_stopping=early_stopping,
                                       seed=seed,
                                       bias=bias)
        
    
    def new_callback(self, iteration, state, fitness, user_data,
                    attempt=0, done=False, curve=None, fitness_evaluations=None):

        if self.extra_callback is None:
            return self._save_state(iteration, state, fitness, user_data,
                    attempt=attempt, done=done, curve=curve, fitness_evaluations=fitness_evaluations)
        
        done = self.extra_callback(iteration, curve, self.extra_callback_info)
        return self._save_state(iteration, state, fitness, user_data,
                    attempt=attempt, done=done, curve=curve, fitness_evaluations=fitness_evaluations)

        

    def _invoke_algorithm(self, algorithm, problem, max_attempts,
                          curve, user_info, additional_algorithm_args=None, **total_args):
        self._current_logged_algorithm_args.update(total_args)
        if additional_algorithm_args is not None:
            self._current_logged_algorithm_args.update(additional_algorithm_args)

        if self.replay_mode() and self._load_pickles():
            return None, None, None

        # arg_text = [get_short_name(v) for v in self._current_logged_algorithm_args.values()]
        self._print_banner('*** Run START ***')
        np.random.seed(self.seed)

        valid_args = [k for k in lk.signature(algorithm).parameters]
        args_to_pass = {k: v for k, v in total_args.items() if k in valid_args}

        self._start_run_timing()
        problem.reset()
        ret = algorithm(problem=problem,
                        max_attempts=max_attempts,
                        curve=curve,
                        random_state=self.seed,
                        state_fitness_callback=self.new_callback,
                        callback_user_info=user_info,
                        **args_to_pass)
        self._print_banner('*** Run END ***')
        self._curve_base = len(self._fitness_curves)
        return ret
    

    def run_one_experiment_(self, algorithm, total_args, **params):
        if self._extra_args is not None and len(self._extra_args) > 0:
            params = {**params, **self._extra_args}

        total_args.update(params)
        total_args.pop('problem')
        user_info = [(k, v) for k, v in total_args.items()]

        return self._invoke_algorithm(algorithm=algorithm,
                                      curve=self.generate_curves,
                                      user_info=user_info,
                                      additional_algorithm_args=total_args,
                                      **params)
