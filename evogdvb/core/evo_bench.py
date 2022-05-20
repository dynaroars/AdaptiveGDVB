import copy
import numpy as np

from enum import Enum, auto

from fractions import Fraction as F
from pathlib import Path

from gdvb.core.verification_benchmark import VerificationBenchmark

from .evo_step import EvoStep

from gdvb.plot.pie_scatter import PieScatter2D


class EvoBench:
    class EvoState(Enum):
        Explore = auto()
        Refine = auto()

    def __init__(self, seed_benchmark):
        self.logger = seed_benchmark.settings.logger
        self.seed_benchmark = seed_benchmark
        self.benchmark_name = seed_benchmark.settings.name
        self.dnn_configs = seed_benchmark.settings.dnn_configs
        # TODO: only support one verifier at a time
        self.verifier = list(seed_benchmark.settings.verification_configs['verifiers'].values())[0][0]
        self.evo_configs = seed_benchmark.settings.evolutionary
        self.evo_params = self.evo_configs['parameters']
        self.explore_iter = self.evo_configs['explore_iterations']
        self.refine_iter = self.evo_configs['refine_iterations']
        assert len(self.evo_params) == 2
        self._init_parameters()

    def _init_parameters(self):
        self.state = self.EvoState.Explore
        self.benchmarks = []
        self.pivots_ua = {}  # under-approximation
        self.pivots_oa = {}  # over-approximation
        self.res = {}
        self.refine_iterations = 3

        for p in self.evo_params:
            self.pivots_ua[p] = None
            self.pivots_oa[p] = None

    def run(self):
        initial_step = EvoStep(self.seed_benchmark, self.evo_params, EvoStep.Direction.Both, 0)
        self.benchmarks += [initial_step]

        # explore
        homework = [initial_step]
        while homework:
            evo_step = homework.pop(0)
            evo_step.forward()
            evo_step.evaluate()
            evo_step.plot()

            self.collect_res(evo_step)
            self.plot(evo_step)

            if evo_step.direction == EvoStep.Direction.Both:
                next_evo_step = self.evolve(evo_step, EvoStep.Direction.Up)
                if next_evo_step:
                    self.benchmarks += [next_evo_step]
                    homework += [next_evo_step]
                next_evo_step = self.evolve(evo_step, EvoStep.Direction.Down)
                if next_evo_step:
                    self.benchmarks += [next_evo_step]
                    homework += [next_evo_step]
            elif evo_step.direction == EvoStep.Direction.Up:
                next_evo_step = self.evolve(evo_step, EvoStep.Direction.Up)
                if next_evo_step:
                    self.benchmarks += [next_evo_step]
                    homework += [next_evo_step]
            elif evo_step.direction == EvoStep.Direction.Down:
                next_evo_step = self.evolve(evo_step, EvoStep.Direction.Down)
                if next_evo_step:
                    self.benchmarks += [next_evo_step]
                    homework += [next_evo_step]
            else:
                raise ValueError(f'Unknown evolve direction: {evo_step.direction}')

        self.logger.info('Exploration finished successfully!')
        exit()

        # clean previous benchmark results for refinement phase
        self.res = None
        self.res_nb_solved = None

        # refine
        while homework:
            evo_step = homework.pop(0)
            self.refine_iterations -= 1
            if self.refine_iterations == 0:
                next_ca_configs = None

        self.logger.info('Refinement finished successfully!')

    def evolve(self, evo_step, direction: EvoStep.Direction):
        ca_configs = evo_step.benchmark.ca_configs

        # A1: Exploration State
        # 1) for all factors, check if exploration is needed
        # 2) for all factors, check if configred bounds are reached
        # if 1) or 2), go to A2: Refinement state
        if self.state == self.EvoState.Explore:
            next_ca_configs = self.explore(evo_step, direction)
            goon = not self.check_same_ca_configs(ca_configs, next_ca_configs)

        # A2 : Refinement State
        # 1) check starting point based on observations from the exploration states
        # 2) prune overly easy and overly hard levels
        # 3) stop when necessary, go to analyze state
        elif self.state == self.EvoState.Refine:
            next_ca_configs = self.refine(evo_step)
            goon = not self.check_same_ca_configs(ca_configs, next_ca_configs)
        else:
            raise ValueError(f'Unknown evolve state: {self.state}')

        if goon:
            next_benchmark = VerificationBenchmark(f'{self.benchmark_name}_{evo_step.iteration}_{evo_step.direction}',
                                                   self.dnn_configs,
                                                   next_ca_configs,
                                                   evo_step.benchmark.settings)

            next_evo_step = EvoStep(next_benchmark,
                                    self.evo_params,
                                    direction,
                                    evo_step.iteration+1)
        else:
            next_evo_step = None

        return next_evo_step

    def explore(self, evo_step: EvoStep, direction: EvoStep.Direction):
        actions = self.update_pivots(evo_step)

        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_next = copy.deepcopy(ca_configs)

        print(f'Evo state: {self.state}; Actions: {list(actions)}')

        print('UA\t', self.pivots_ua)
        print('OA\t', self.pivots_oa)

        # check if no scales are needed for the entire slice
        if all(x == 1.0 for x in actions.flatten().tolist()):
            print('GOTO: Refine due to same action compared to last step.')
            self.state = self.EvoState.Refine
        # explore
        else:
            parameters_lower_bounds = self.evo_configs['parameters_lower_bounds']
            parameters_upper_bounds = self.evo_configs['parameters_upper_bounds']

            for i, f in enumerate(evo_step.factors):
                f = copy.deepcopy(f)
                start = f.start * F(actions[i][0])
                end = f.end * F(actions[i][1])

                # check hard bounds from evo configs
                if f.type in parameters_lower_bounds:
                    start = max(start, F(parameters_lower_bounds[f.type]))
                if f.type in parameters_upper_bounds:
                    end = min(end, F(parameters_upper_bounds[f.type]))

                # skip factor-level modification if start >= end
                if start >= end:
                    self.logger.warn(f'START > END!!! NO MODIFICATION TO FACTOR: {f.type}')
                    continue

                f.set_start_end(start, end)

                start, end, levels = f.get()
                ca_configs_next['parameters']['level'][f.type] = levels
                ca_configs_next['parameters']['range'][f.type] = [start, end]

        return ca_configs_next

    def update_pivots(self, evo_step):
        nb_property = evo_step.benchmark.ca_configs['parameters']['level']['prop']
        #nb_levels = np.array([x.nb_levels for x in evo_step.factors])
        res = evo_step.nb_solved[self.verifier]
        # TEST res
        # res = np.array([[5,5,5],[5,5,1],[3,1,5]])
        max_value = nb_property
        # min_value = 0
        print(res)

        max_ids = np.array(np.where(max_value == res)).T
        ua_candidates = []

        # Estimate verification boundary under-approximation
        for max_id in max_ids:
            ids = np.array(list(np.ndindex(*(max_id+1))))

            issubset = True
            for x in ids:
                if x.tolist() not in max_ids.tolist():
                    issubset = False
                    break

            # print('Subject:', max_id, 'Result:', issubset)
            if issubset:
                ua_candidates += [max_id]

        # expand lower
        if len(ua_candidates) == 0:
            pivot_ua_id = None
        # pivot lb found
        else:
            levels = np.array([x.explicit_levels for x in evo_step.factors], dtype=F)

            ua_candidates_real_level = []
            for mc in ua_candidates:
                temp = []
                for d_i, x in enumerate(mc):
                    temp += [levels[d_i][x]]
                ua_candidates_real_level += [np.array(temp)]
            ua_candidates_real_level_prod = [np.prod(x) for x in ua_candidates_real_level]
            print(ua_candidates_real_level)
            print(ua_candidates_real_level_prod)
            pivot_ua_id = np.argmax(ua_candidates_real_level_prod)
            print(pivot_ua_id)
            # for i, x in enumerate(ua_candidates_real_level[pivot_ua_id]):
            #    print(i, x)

        lb_cuts = []
        ub_cuts = []
        # max_cut & min_cut
        for i, f in enumerate(evo_step.factors):
            axes = list(range(len(self.evo_params)))
            axes.remove(i)
            axes = [i] + axes
            raw = res.transpose(axes)

            lb_cut = [np.all(x == nb_property) for x in raw]
            print(lb_cut)
            lb_cut = lb_cut.index(True) if True in lb_cut else None
            print(lb_cut)
            lb_cuts += [lb_cut]
            ub_cut = [np.all(x == 0) for x in reversed(raw)]
            print(ub_cut)
            ub_cut = ub_cut.index(True) if True in ub_cut else None
            print(ub_cut)
            ub_cuts += [ub_cut]

            print(f'[{f.type}] lb_cut: {lb_cut}; ub_cut: {ub_cut}.')

        actions = np.zeros([len(self.evo_params), 2])
        for i, f in enumerate(evo_step.factors):

            # update under-approximation pivot
            # ua pivot not found: scale down
            if pivot_ua_id is None:
                actions[i][0] = self.evo_configs['deflation_rate']
            # ua pivot found: check if ua pivot on border
            else:
                # update ua pivot
                if self.pivots_ua[f.type] is None:
                    self.pivots_ua[f.type] = ua_candidates_real_level[pivot_ua_id][i]
                else:
                    self.pivots_ua[f.type] = max(self.pivots_ua[f.type], ua_candidates_real_level[pivot_ua_id][i])

                # T: ua pivot too small -> scale up
                if lb_cuts[i] is not None:
                    actions[i][0] = self.evo_configs['inflation_rate']
                # F: ua pivot not too small -> don't scale
                else:
                    actions[i][0] = 1

            # update over-approximation pivot
            # oa pivot not found: scale up
            if ub_cuts[i] is None:
                actions[i][1] = self.evo_configs['inflation_rate']
            # oa pivot found: check if oa pivot on border
            else:
                # update oa pivot
                if self.pivots_oa[f.type] is None:
                    print(f.explicit_levels[f.nb_levels-1-ub_cuts[i]])
                    self.pivots_oa[f.type] = f.explicit_levels[f.nb_levels-1-ub_cuts[i]]
                else:
                    self.pivots_oa[f.type] = min(self.pivots_oa[f.type], f.explicit_levels[f.nb_levels-1-ub_cuts[i]])
                # T: oa pivot too large -> scale down
                if ub_cuts[i]+1 == f.nb_levels:
                    actions[i][1] = self.evo_configs['deflation_rate']
                # F: oa pivot not too large -> don't scale
                else:
                    actions[i][1] = 1

        return actions

    def refine(self, evo_step):
        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_next = copy.deepcopy(ca_configs)
        arity = self.evo_configs['refine_arity']

        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)

            start = self.pivots_ua[f.type]
            end = self.pivots_oa[f.type]

            if start is None or end is None:
                raw = self.res_nb_solved[list(self.res_nb_solved)[0]]
                all_levels = set(x[i] for x in raw)
                min_level = min(all_levels)
                max_level = max(all_levels)

                print(all_levels, min_level, max_level)

                pivot_min_candidates = [min_level]
                pivot_max_candidates = [max_level]
                for l in all_levels:
                    min_pass = True
                    max_pass = True
                    for x in raw:
                        if x[i] <= l:
                            if raw[x] == ca_configs['parameters']['level']['prop'] or raw[x] >= x[i]:
                                min_pass = False
                        if x[i] >= l:
                            if raw[x] != 0:
                                max_pass = False

                    if min_pass:
                        pivot_min_candidates += [l]
                    if max_pass:
                        pivot_max_candidates += [l]

                print("pivot_min_candidates: ", pivot_min_candidates)
                print("pivot_max_candidates: ", pivot_max_candidates)

                if start is None:
                    start = max(pivot_min_candidates)
                if end is None:
                    end = min(pivot_max_candidates)
            print(start)
            print(end)
            assert start is not None and end is not None and start <= end, f'!Wrong start/end. Start: {start}, End: {end}'

            f.set_start_end(start, end)
            f.subdivision(arity)

            start, end, levels = f.get()
            ca_configs_next['parameters']['level'][f.type] = levels
            ca_configs_next['parameters']['range'][f.type] = [start, end]

        return ca_configs_next

    def check_same_ca_configs(self, this, that):
        res = []
        for p in self.evo_params:
            this_start = F(this['parameters']['range'][p][0])
            that_start = F(that['parameters']['range'][p][0])
            this_end = F(this['parameters']['range'][p][1])
            that_end = F(that['parameters']['range'][p][1])
            this_level = F(this['parameters']['level'][p])
            that_level = F(that['parameters']['level'][p])

            res += [this_start == that_start]
            res += [this_end == that_end]
            res += [this_level == that_level]

        # print(res, all(x for x in res))
        return all(x for x in res)

    def collect_res(self, evo_step):
        if not self.res:
            self.res = {v: {} for v in evo_step.answers}
            self.res_nb_solved = {v: {} for v in evo_step.answers}

        levels = tuple(f.explicit_levels for f in evo_step.factors)

        # TODO : WTF???? how to separate ndarray _,_ = np.xxx(x)???
        ids = np.array(np.meshgrid(
            levels[0], levels[1])).T.reshape(-1, len(self.evo_params))

        data = list(evo_step.answers.values())[0]
        data = data.reshape(-1, data.shape[-1])

        data2 = list(evo_step.nb_solved.values())[0]
        data2 = data2.reshape(-1, 1)

        # verifier = list(evo_step.answers)[0]
        verifier = self.verifier
        for i, id in enumerate(ids):
            self.res[verifier][tuple(id)] = data[i]
            self.res_nb_solved[verifier][tuple(id)] = data2[i]

    # plot two factors with properties: |F| = 3
    # TODO: update plotter to accept more than two factors
    def plot(self, evo_step):
        if len(self.evo_params) == 2:
            labels = [x for x in self.evo_params]
            ticks = {x: set() for x in self.evo_params}

            #verifier = list(self.benchmarks[0].answers)[0]
            verifier = self.verifier
            ticks = np.array([list(x) for x in self.res[verifier].keys()], dtype=np.float32)
            data = np.array([x for x in self.res[verifier].values()], dtype=np.float32)

            # print(self.evo_params[0], set(sorted(np.array([list(x) for x in self.res[verifier].keys()])[:, 0].tolist())))
            # print(self.evo_params[1], set(sorted(np.array([list(x) for x in self.res[verifier].keys()])[:, 1].tolist())))

            ticks_f1 = ticks[:, 0].tolist()
            ticks_f2 = ticks[:, 1].tolist()

            labels_f1 = labels[0]
            labels_f2 = labels[1]

            pie_scatter = PieScatter2D(data)
            pie_scatter.draw_with_ticks(ticks_f1, ticks_f2, labels_f1, labels_f2)

            pdf_dir = f'{self.seed_benchmark.settings.root}/figures/'
            Path(pdf_dir).mkdir(parents=True, exist_ok=True)
            pie_scatter.save(f'{pdf_dir}/all_{evo_step.iteration}_{evo_step.direction}.png')

            pie_scatter.draw_with_ticks(ticks_f1, ticks_f2, labels_f1, labels_f2, x_log_scale=True, y_log_scale=True)
            pie_scatter.save(f'{pdf_dir}/all_log_{evo_step.iteration}_{evo_step.direction}.png')

        else:
            # plot two factors with properties: |F| >= 3
            # TODO: update plotter to accept more than two factors
            raise NotImplementedError
