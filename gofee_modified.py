""" Definition of GOFEE class.
"""

import numpy as np
import pickle
from os.path import isfile

from ase import Atoms
from ase.io import read, write, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.dftb import Dftb

from gofee.surrogate import GPR
from gofee.population import Population
from gofee.utils import array_to_string
from gofee.parallel_utils import split, parallel_function_eval

from gofee.bfgslinesearch_constrained import relax

from gofee.candidates import CandidateGenerator
from gofee.candidates import RattleMutation

from mpi4py import MPI
world = MPI.COMM_WORLD

import traceback
import sys
from os import path

from time import time

class GOFEE():
    """
    GOFEE global structure search method.
        
    Parameters:

    structures: Atoms-object, list of Atoms-objects or None
        In initial structures from which to start the sesarch.
        If None, the startgenerator must be supplied.
        If less than Ninit structures is supplied, the remaining
        ones are generated using the startgenerator or by rattling
        the supplied structures, depending on wether the
        startgenerator is supplied.

    calc: ASE calculator
        Specifies the energy-expression
        with respect to which the atomic coordinates are
        globally optimized.

    gpr: GPR object
        The Gaussian Process Regression model used as the
        surrogate model for the Potential energy surface.
    
    startgenerator: Startgenerator object
        Used to generate initial random
        structures. Must be supplied if structures if structues=None.
        (This is the recommended way to initialize the search.)

    candidate_generator: OperationSelector object
        Object used to generate new candidates.

    trajectory: str
        Name of trajectory to which all structures,
        evaluated during the search, is saved.

    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.

    kappa: float
        How much to weigh predicted uncertainty in the acquisition
        function.

    max_steps: int
        Number of search steps.

    Ninit: int
        Number of initial structures. If len(structures) <
        Ninit, the remaining structures are generated using the
        startgenerator (if supplied) or by rattling the supplied
        'structures'.

    max_relax_dist: float
        Max distance (in Angstrom) that an atom is allowed to
        move during surrogate relaxation.

    Ncandidates: int
        Number of new cancidate structures generated and
        surrogate-relaxed in each search iteration.

    population_size: int
        Maximum number of structures in the population.

    dualpoint: boolean
        Whether to use dualpoint evaluation or not.

    min_certainty: float
        Max predicted uncertainty allowed for structures to be
        considdered for evaluation. (in units of the maximum possible
        uncertainty.)

    position_constraint: OperationConstraint object
        Enforces constraints on the positions of the "free" atoms
        in the search. The constraint is enforces both during
        mutation/crossover operations and during surrogate-relaxation.

    restart: str
        Filename for restart file.
    """
    def __init__(self, structures=None,
                 calc=None,
                 gpr=None,
                 startgenerator=None,
                 candidate_generator=None,
                 kappa=2,
                 max_steps=200,
                 Ninit=10,
                 max_relax_dist=4,
                 Ncandidates=30,
                 population_size=10,
                 dualpoint=True,
                 min_certainty=0.7,
                 position_constraint=None,
                 trajectory='structures.traj',
                 logfile='search.log',
                 restart='restart.pickl',
                 old_trajectory=None, #new input variable, SAM 22/07
                 estd_thr=0): #new input variable, SAM 22/07
        
        if structures is None:
            assert startgenerator is not None
            self.structures = None
        else:
            if isinstance(structures, Atoms):
                self.structures = [structures]
            elif isinstance(structures, list):
                assert isinstance(structures[0], Atoms)
                self.structures = structures
            elif isinstance(structures, str):
                self.structures = read(structures, index=':')
        
        if calc is not None:
            self.calc = calc
        else:
            assert structures is not None
            calc = structures[0].get_calculator()
            assert calc is not None and not isinstance(calc, SinglePointCalculator)
            self.calc = calc

        if startgenerator is None:
            assert structures is not None
            self.startgenerator = None
        else:
            self.startgenerator = startgenerator

        # Determine atoms to optimize
        if startgenerator is not None:
            self.n_to_optimize = len(self.startgenerator.stoichiometry)
        else:
            self.n_to_optimize = len(self.structures[0])
            for constraint in self.structures[0].constraints:
                if isinstance(constraint, FixAtoms):
                    indices_fixed = constraint.get_indices()
                    self.n_to_optimize -= len(indices_fixed)
                    break
        
        # Set up candidate-generator if not supplied
        if candidate_generator is not None:
            self.candidate_generator = candidate_generator
        else:
            rattle = RattleMutation(self.n_to_optimize,
                                    Nrattle=3,
                                    rattle_range=4)
            self.candidate_generator = CandidateGenerator([1.0],[rattle])

        # Define parallel communication
        self.comm = world.Dup()  # Important to avoid mpi-problems from call to ase.parallel in BFGS
        self.master = self.comm.rank == 0

        self.kappa = kappa
        #self.max_steps = max_steps
        self.Ninit = Ninit
        self.max_relax_dist = max_relax_dist
        self.Ncandidates = Ncandidates
        self.dualpoint = dualpoint
        self.min_certainty = min_certainty
        self.position_constraint = position_constraint
        self.restart = restart
        self.estd_thr = estd_thr
        self.dataset = None #2022/08/12 SAM: number of data from old_trajectory
        
        #New lines, SAM 22/07/04
        if old_trajectory is not None:
            self.max_steps = max_steps + 50
        else:
            self.max_steps = max_steps
        #New lines, SAM
        
        # Add position-constraint to candidate-generator
        self.candidate_generator.set_constraints(position_constraint)

        if isinstance(trajectory, str):
            self.trajectory = Trajectory(filename=trajectory, mode='a', master=self.master)
            if self.restart:
                self.traj_name = trajectory
        
        ### New lines, SAM 22/07
        if isinstance(old_trajectory, str):
            self.old_trajectory = Trajectory(filename=old_trajectory, mode='a', master=self.master)
            if self.restart:
                self.old_traj_name = old_trajectory
        ### New lines, SAM 22/07
        
        if not self.master:
            logfile = None
        elif isinstance(logfile, str):
            if logfile == "-":
                logfile = sys.stdout
            else:
                logfile = open(logfile, "a")
        self.logfile = logfile
        self.log_msg = ''

        if restart is None or not path.exists(restart):
            self.initialize()

            if gpr is not None:
                self.gpr = gpr
            else:
                self.gpr = GPR(template_structure=self.structures[0])
            
            self.population = Population(population_size=population_size, gpr=self.gpr, similarity2equal=0.9999)
        ### New lines, SAM 22/07: To read the old trajectory and reuse surrogate
        elif old_trajectory is not None:
            self.get_initial_structures()
            self.old_read()
            self.comm.barrier()
            self.steps = 0
            
            # Initialize population
            self.population = Population(population_size=population_size, gpr=self.gpr, similarity2equal=0.9999)
        ### New line, SAM 22/07
        elif trajectory is not None:
            self.read()
            self.comm.barrier()


    def initialize(self):
        self.get_initial_structures()
        self.steps = 0

    def get_initial_structures(self):
        """Method to prepare the initial structures for the search.
        
        The method makes sure that there are atleast self.Ninit
        initial structures.
        These structures are first of all the potentially supplied
        structures. If more structures are required, these are
        generated using self.startgenerator (if supplied), otherwise
        they are generated by heavily rattling the supplied structures.
        """
        
        # Collect potentially supplied structures.
        if self.structures is not None:
            for a in self.structures:
                a.info = {'origin': 'PreSupplied'}
        else:
            self.structures = []
        
        Nremaining = self.Ninit - len(self.structures)
        
        if Nremaining > 0 and self.startgenerator is None:
            # Initialize rattle-mutation for all atoms.
            rattle = RattleMutation(self.n_to_optimize,
                                    Nrattle=self.n_to_optimize,
                                    rattle_range=2)

        # Generation of remaining initial-structures (up to self.Ninit).
        for i in range(Nremaining):
            if self.startgenerator is not None:
                a = self.startgenerator.get_new_candidate()
            else:
                # Perform two times rattle of all atoms.
                a0 = self.structures[i % len(self.structures)]
                a = rattle.get_new_candidate([a])
                a = rattle.get_new_candidate([a])
            self.structures.append(a)
                
    def evaluate_initial_structures(self):
        """ Evaluate energies and forces of all initial structures
        (self.structures) that have not yet been evaluated.
        """
        structures_init = []
        for a in self.structures:
            calc = a.get_calculator()
            if isinstance(calc, SinglePointCalculator):
                if 'energy' in calc.results and 'forces' in calc.results:
                    # Write without evaluating.
                    structures_init.append(a)
                    continue
            a = self.evaluate(a) #modified, SAM 22/07. Previous: a = self.evaluate(a)
            structures_init.append(a)

        self.gpr.memory.save_data(structures_init)
        self.population.add(structures_init)

    def run(self):
        """ Method to run the search.
        """
        if self.steps == 0:
            self.evaluate_initial_structures()
            
            self.log_msg += "#### Training result from previous and current initial data: #####\n "
            self.train_surrogate() # new line, SAM 22/07
               
        while self.steps < self.max_steps:
            self.log_msg += (f"\n##### STEPS: {self.steps} #####\n\n")
            t0 = time()
            self.update_population()
            t1 = time()
            unrelaxed_candidates = self.generate_new_candidates()
            t2 = time()
            relaxed_candidates = self.relax_candidates_with_surrogate(unrelaxed_candidates)
            t3 = time()
            kappa = self.kappa
            a_add = []
            for _ in range(5):
                try:
                    anew = self.select_with_acquisition(relaxed_candidates, kappa)
                    
                    if anew.info['key_value_pairs']['Epred_std'] > self.estd_thr:
                        anew = self.evaluate(anew)
                        a_add.append(anew)
                        self.gpr.memory.save_data(a_add)
                    else:
                        self.write(anew)
                        
                    if self.dualpoint:
                        if anew.info['key_value_pairs']['Epred_std'] > self.estd_thr:
                            adp = self.get_dualpoint(anew)
                            adp = self.evaluate(adp)
                            a_add.append(adp)
                            self.gpr.memory.save_data(a_add)
                        else:
                            self.write(anew)
                    break
                except Exception as err:
                    kappa /=2
                    if self.master:
                        traceback.print_exc(file=sys.stderr)
                      
            t4 = time()
            # New lines, SAM 22/07: To skip training.
            if anew.info['key_value_pairs']['Epred_std'] > self.estd_thr:            
                self.train_surrogate()
            else:
                self.log_msg += "DFT evaluation and training is skipped\n"
            #New lines, SAM 22/07
            
            # log timing
            self.log_msg += "Timing:\n"
            self.log_msg += f"{'Relax pop.':12}{'Make cands.':12}{'Relax cands.':15}{'Evaluate':16}{'Training':12}\n" #Modified, SAM 22/07. Training is moved to end. 
            self.log_msg += f"{t1-t0:<12.2e}{t2-t1:<12.2e}{t3-t2:<15.2e}{t4-t3:<16.2e}{time()-t4:<12.2e}\n\n"

            # Add structure to population
            if anew.info['key_value_pairs']['Epred_std'] > self.estd_thr:
                index_lowest = np.argmin([a.get_potential_energy() for a in a_add])
                self.population.add([a_add[index_lowest]])

            # Save search state
            self.save_state()
            
            self.log_msg += (f"Prediction:\nenergy = {anew.info['key_value_pairs']['Epred']:.5f}eV,  energy_std = {anew.info['key_value_pairs']['Epred_std']:.5f}eV\n")
            if anew.info['key_value_pairs']['Epred_std'] > self.estd_thr:
                self.log_msg += (f"E_true:\n{array_to_string([a.get_potential_energy() for a in a_add], unit='eV')}\n\n")
            #self.log_msg += (f"E_true: {[a.get_potential_energy() for a in a_add]}\n")
            self.log_msg += (f"Energy of population:\n{array_to_string([a.get_potential_energy() for a in self.population.pop], unit='eV')}\n")
            self.log_msg += (f"Max force of ML-relaxed population:\n{array_to_string([(a.get_forces()**2).sum(axis=1).max()**0.5 for a in self.population.pop_MLrelaxed], unit='eV/A')}\n")
            
            self.log()
            self.steps += 1
            
    def get_dualpoint(self, a, lmax=0.10, Fmax_flat=5):
        """Returns dual-point structure, i.e. the original structure
        perturbed slightly along the forces.
        
        lmax: The atom with the largest force will be displaced by
        this distance
        
        Fmax_flat: maximum atomic displacement. is increased linearely
        with force until Fmax = Fmax_flat, after which it remains
        constant as lmax.
        """
        F = a.get_forces()
        a_dp = a.copy()

        # Calculate and set new positions
        Fmax = np.sqrt((F**2).sum(axis=1).max())
        pos_displace = lmax * F*min(1/Fmax_flat, 1/Fmax)
        pos_dp = a.positions + pos_displace
        a_dp.set_positions(pos_dp)
        return a_dp

    def generate_new_candidates(self):
        """Method to generate a self.Ncandidates new candidates
        by applying the operations defined in self.candidate_generator
        to the structures currently in the population.
        The tasks are parrlelized over all avaliable cores.
        """
        Njobs = self.Ncandidates
        task_split = split(Njobs, self.comm.size)
        def func1():
            return [self.generate_candidate() for i in task_split[self.comm.rank]]
        candidates = parallel_function_eval(self.comm, func1)
        return candidates

    def relax_candidates_with_surrogate(self, candidates):
        """ Method to relax new candidates using the
        surrogate-model.
        The tasks are parrlelized over all avaliable cores.
        """
        Njobs = self.Ncandidates
        task_split = split(Njobs, self.comm.size)
        def func2():
            return [self.surrogate_relaxation(candidates[i], Fmax=0.1, steps=200, kappa=self.kappa)
                    for i in task_split[self.comm.rank]]
        relaxed_candidates = parallel_function_eval(self.comm, func2)
        relaxed_candidates = self.certainty_filter(relaxed_candidates)
        relaxed_candidates = self.population.pop_MLrelaxed + relaxed_candidates
        
        return relaxed_candidates

    def generate_candidate(self):
        """ Method to generate new candidate.
        """
        parents = self.population.get_structure_pair()
        a_mutated = self.candidate_generator.get_new_candidate(parents)
        return a_mutated

    def surrogate_relaxation(self, a, Fmax=0.1, steps=200, kappa=None):
        """ Method to carry out relaxations of new candidates in the
        surrogate potential.
        """
        calc = self.gpr.get_calculator(kappa)
        a_relaxed = relax(a, calc, Fmax=Fmax, steps_max=steps,
                          max_relax_dist=self.max_relax_dist,
                          position_constraint=self.position_constraint)
        # Evaluate uncertainty
        E, Estd = self.gpr.predict_energy(a_relaxed, eval_std=True)

        # Save prediction in info-dict
        a_relaxed.info['key_value_pairs']['Epred'] = E
        a_relaxed.info['key_value_pairs']['Epred_std'] = Estd
        a_relaxed.info['key_value_pairs']['kappa'] = self.kappa
        
        return a_relaxed
        
    def certainty_filter(self, structures):
        """ Method to filter away the most uncertain surrogate-relaxed
        candidates, which might otherewise get picked for first-principles
        evaluation, based on the very high uncertainty alone.
        """
        certainty = np.array([a.info['key_value_pairs']['Epred_std']
                              for a in structures]) / np.sqrt(self.gpr.K0)
        min_certainty = self.min_certainty
        for _ in range(5):
            filt = certainty < min_certainty
            if np.sum(filt.astype(int)) > 0:
                structures = [structures[i] for i in range(len(filt)) if filt[i]]
                break
            else:
                min_certainty = min_certainty + (1-min_certainty)/2
        return structures

    def update_population(self):
        """ Method to update the population with the new pirst-principles
        evaluated structures.
        """
        Njobs = len(self.population.pop)
        task_split = split(Njobs, self.comm.size)
        func = lambda: [self.surrogate_relaxation(self.population.pop[i],
                                                  Fmax=0.001, steps=200, kappa=None)
                        for i in task_split[self.comm.rank]]
        self.population.pop_MLrelaxed = parallel_function_eval(self.comm, func)
        Fmax_pop_relaxed = [(a.get_forces()**2).sum(axis=1).max()**0.5
                                  for a in self.population.pop_MLrelaxed]

    def train_surrogate(self):
        """ Method to train the surrogate model.
        The method only performs hyperparameter optimization every 
        ten training instance, as carrying out the hyperparameter
        optimization is significantly more expensive than the basic
        training.
        """
        # Train #2022/08/12: SAM, reduce hyperparameter optimization for run with prev. surrogate. 
        if self.old_trajectory is not None:
            if (self.steps % 10) == 0:
                self.gpr.optimize_hyperparameters(comm=self.comm)
                self.log_msg += (f"lml: {self.gpr.lml}\n")
                self.log_msg += (f"kernel optimized:\nTheta = {[f'{x:.2e}' for x in np.exp(self.gpr.kernel.theta)]}\n\n")
            else:
                self.gpr.train()
                self.log_msg += (f"kernel fixed:\nTheta = {[f'{x:.2e}' for x in np.exp(self.gpr.kernel.theta)]}\n\n")
        else:
            if self.steps < 50 or (self.steps % 10) == 0:
                self.gpr.optimize_hyperparameters(comm=self.comm)
                self.log_msg += (f"lml: {self.gpr.lml}\n")
                self.log_msg += (f"kernel optimized:\nTheta = {[f'{x:.2e}' for x in np.exp(self.gpr.kernel.theta)]}\n\n")
            else:
                self.gpr.train()
                self.log_msg += (f"kernel fixed:\nTheta = {[f'{x:.2e}' for x in np.exp(self.gpr.kernel.theta)]}\n\n")

    def select_with_acquisition(self, structures, kappa):
        """ Method to select single most "promizing" candidate 
        for first-principles evaluation according to the acquisition
        function min(E-kappa*std(E)).
        """
        Epred = np.array([a.info['key_value_pairs']['Epred']
                          for a in structures])
        Epred_std = np.array([a.info['key_value_pairs']['Epred_std']
                              for a in structures])
        acquisition = Epred - kappa*Epred_std
        index_select = np.argmin(acquisition)
        return structures[index_select]
    
    def evaluate(self, a):
        a = self.comm.bcast(a, root=0)
        a.wrap()

        if isinstance(self.calc, Dftb):
            if self.master:
                try:
                    a.set_calculator(self.calc)
                    E = a.get_potential_energy()
                    F = a.get_forces()
                    results = {'energy': E, 'forces': F}
                    calc_sp = SinglePointCalculator(a, **results)
                    a.set_calculator(calc_sp)
                    success = True
                except:
                    success = False
            else:
                success = None
            success = self.comm.bcast(success, root=0)
            if success == False:
                raise RuntimeError('DFTB evaluation failed')
            a = self.comm.bcast(a, root=0)
        else:
            a.set_calculator(self.calc)
            E = a.get_potential_energy()
            F = a.get_forces()
            results = {'energy': E, 'forces': F}
            calc_sp = SinglePointCalculator(a, **results)
            a.set_calculator(calc_sp)

        self.write(a)

        return a

    def write(self, a):
        """ Method for writing new evaluated structures to file.
        """
        if self.trajectory is not None:
            self.trajectory.write(a)

    def dump(self, data):
        """ Method to save restart-file used if the search is
        restarted from some point in the search. 
        """
        if self.comm.rank == 0 and self.restart is not None:
            pickle.dump(data, open(self.restart, "wb"), protocol=2)

    def save_state(self):
        """ Saves the current state of the search, so the search can
        be continued after having finished or stoped prematurely.
        """
        self.dump((self.steps,
                   self.population,
                   self.gpr.kernel.theta,
                   np.random.get_state()))

    def read(self):
        """ Method to restart a search from the restart-file and the
        trajectory-file containing all structures evaluated so far.
        """
        self.steps, self.population, theta, random_state = pickle.load(open(self.restart, "rb"))
        np.random.set_state(random_state)
        training_structures = read(self.traj_name, index=':')

        # Salvage GPR model
        self.gpr = GPR(template_structure=training_structures[0])
        self.gpr.memory.save_data(training_structures)
        self.gpr.kernel.theta = theta
    
    # New function, SAM 22/07: To read restart and old_trajectory
    def old_read(self):
        self.steps, self.population, theta, random_state = pickle.load(open(self.restart, "rb"))
        np.random.set_state(random_state)
        training_structures = read(self.old_traj_name, index=':')

        # Salvage GPR model
        self.gpr = GPR(template_structure=training_structures[0])
        self.gpr.memory.save_data(training_structures)
        self.gpr.kernel.theta = theta
        
        # Number of prev. dataset
        self.dataset = len(training_structures) #2022/08/12 SAM: number of data from old_trajectory
    # New function, SAM 22/07. 
        
    def log(self):
        if self.logfile is not None:
            if self.steps == 0:
                msg1 = "GOFEE modified version\n"
                self.logfile.write(msg1)
                if self.old_trajectory is not None: #2022/08/12 SAM: number of data from old_trajectory
                    msg2 = (f"Number of structures in old_trajectory = {self.dataset} \n\n")
                    self.logfile.write(msg2)

            self.logfile.write(self.log_msg)
            self.logfile.flush()
        self.log_msg = ''


