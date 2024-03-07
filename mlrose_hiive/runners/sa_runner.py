import mlrose_hiive
from mlrose_hiive.decorators import short_name
from mlrose_hiive.runners._runner_base import _RunnerBase
import numpy as np
import inspect as lk

"""
Example usage:

    experiment_name = 'example_experiment'
    problem = TSPGenerator.generate(seed=SEED, number_of_cities=22)
    
    # note that you can also initialize a temperature_list this way
    # temperature_list = [mlrose_hiive.GeomDecay(init_temp=t, decay=d) for (t, d) in [(1, 0.99), (1e2, 0.999)]]
    # if you use this form, the decay_list parameter is ignored.

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                  decay_list=[mlrose_hiive.GeomDecay])

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()                  
"""


@short_name('sa')
class SARunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, temperature_list, decay_list=None,
                 max_attempts=500, generate_curves=True, extra_callback=None, extra_callback_info=None, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.use_raw_temp = True
        self.temperature_list = temperature_list
        self.extra_callback = extra_callback
        self.extra_callback_info = extra_callback_info

        if all([np.isscalar(x) for x in temperature_list]):
            if decay_list is None:
                decay_list = [mlrose_hiive.GeomDecay]
            self.decay_list = decay_list
            self.use_raw_temp = False

    


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
    


    def run(self):
        temperatures = self.temperature_list if self.use_raw_temp else [d(init_temp=t) for t in self.temperature_list
                                                                        for d in self.decay_list]
        return super().run_experiment_(algorithm=mlrose_hiive.simulated_annealing,
                                       schedule=('Temperature', temperatures))
