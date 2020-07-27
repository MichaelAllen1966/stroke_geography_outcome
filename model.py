"""
Results generated

 0: Number of hospitals
 1: Mean time to thrombolysis
 2: Max time to thrombolysis
 3: Mean time to thrombectomy
 4: Maximum time to thrombectomy
 5: Minimum thrombolysis admissions to any one hospital
 6: Maximum thrombolysis admissions to any one hospital
 7: Minimum thrombectomy admissions to any one hospital
 8: Maximum thrombectomy admissions to any one hospital
 9: Proportion patients within target thrombolysis time
10: Proportion patients attending unit with target first admissions
11: Proportion patients meeting both thrombolysis targets
12: Proportion patients within target thrombectomy time
13: Proportion patients attending unit with target thrombectomy
14: Proportion patients meeting targets both thrombectomy targets
15: Proportion patients meeting all thrombolysis + thrombectomy targets
16: 95th percentile time for thrombolysis
17: 95th percentile time for thrombectomy
18: Total transfers
19: Total transfer time
20: Clinical outcome (good outcomes) with no treatment
21: Clinical outcome (good outcomes) with treatment
22: Additional good outcomes per 1000 admissions
23: Median time to thrombolysis
24: Median time to thrombectomy
25: Minimum clinical outcome 
26: 5th percentile clinical outcome
27: 95th percentile clinical outcome
28: Maximum clinical outcome
"""

# Import general modules
import datetime
import time
import numpy as np
import pandas as pd

# Import class modules
from classes.data import Data
from classes.population import Pop
from classes.score import Score_population
from classes.score_with_diagnostic import Score_population_with_diagnostic
from classes.pareto import Pareto


# from classes.pareto import Pareto

class Model():
    """
    Main algorithm code.
    1) Initialise algorithm object and load data
    """

    def __init__(self):
        """
        Set up algorithm environment.
        Load:
            Global variables
            Underlying data for algorithm:
                List of hospitals
                Patients by LSOA
                Travel matrix from all LSOA to all hospitals
        """
        # Set up class instances (one instance per class)
        self.data = Data()
        self.pop = Pop()
        self.time_of_last_save = 0

        return

    def initialise_population(self):
        """
        This method creates a starting population.
        This may consist of:
          a) a random population,
          b) a loaded population,
          c) both
        """
        self.pop.population = []

        # Generate random population if required
        if self.data.initial_random_population_size > 0:
            self.pop.population = self.pop.create_random_population(
                self.data.initial_random_population_size,
                self.data.hospitals,
                self.data.fix_hospital_status)

        # Process loaded population in required
        if len(self.data.loaded_population) > 0:
            # Combine randome and loaded populatiosn if required
            if len(self.pop.population) > 0:
                self.pop.population = np.vstack((self.data.loaded_population,
                                                 self.pop.population))
            else:
                self.pop.population = self.data.loaded_population

            # Fix hospital status (if required)
            if self.data.fix_hospital_status:
                self.pop.population = self.pop.fix_hospital_status(
                    self.data.hospitals, self.pop.population)

            # Remove non-unique rows
            self.pop.population = np.unique(self.pop.population, axis=0)

        # Remove any rows with no hospitals
        check_hospitals = np.sum(self.pop.population, axis=1) > 0
        self.pop.population = self.pop.population[check_hospitals, :]

        return

    def run_algorithm(self):
        # Create initial population
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              'Loading coffee and biscuits')
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              'Starting generations')

        # Initialise peopulation
        self.initialise_population()

        # Score first popultion

        # Score_population_with_diagnostic

        if self.data.use_diagnostic:
            self.score = Score_population_with_diagnostic(
                self.data, self.pop.population)
        else:
            self.score = Score_population(self.data, self.pop.population)

        # Get pareto front of first population (if required)
        if self.data.apply_pareto_to_starting_population:
            self.pareto_front = Pareto(self.score,
                                       self.data.pareto_scores_used,
                                       self.data.minimum_population_size,
                                       self.data.maximum_population_size,
                                       self.pop)

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              'Generation: Start, Patients in target distance/admissions: '
              '%1.4f, Benefit: %2.1f, Population size %5.0f'
              % (np.max(self.score.results[:, 11]), np.max(
                  self.score.results[:, 22]), self.pop.population.shape[0]))

        for generation in range(self.data.maximum_generations):
            # Add new random population
            new_population_members_required = (
                int(self.pop.population.shape[0] *
                    self.data.proportion_new_random_each_generation) + 1)

            new_population = self.pop.create_random_population(
                new_population_members_required,
                self.data.hospitals,
                self.data.fix_hospital_status)

            # Combine populations before breeding
            self.pop.population = np.vstack((self.pop.population,
                                             new_population))

            # Get new children
            child_population = (self.pop.generate_child_population(
                self.data.max_crossover_points,
                self.data.mutation_rate,
                self.data.hospitals,
                self.data.fix_hospital_status))

            # Combine populations
            self.pop.population = np.vstack((self.pop.population,
                                             child_population))

            # Remove scenarios with no hospitals
            check_hospitals = np.sum(self.pop.population, axis=1) > 0
            self.pop.population = self.pop.population[check_hospitals, :]

            # Remove non-unique rows
            self.pop.population = np.unique(self.pop.population, axis=0)

            # Score popultion
            if self.data.use_diagnostic:
                self.score = Score_population_with_diagnostic(
                    self.data, self.pop.population)
            else:
                self.score = Score_population(self.data, self.pop.population)

            # Get parteo front (updates population and scores)
            self.pareto_front = Pareto(self.score,
                                       self.data.pareto_scores_used,
                                       self.data.minimum_population_size,
                                       self.data.maximum_population_size,
                                       self.pop)

            # save latest results if more than 120 min since last save
            if time.time() - self.time_of_last_save > (15*60):
                self.time_of_last_save = time.time()
                self.save_results()

            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                  'Generation: %5.0f, Patients in target distance/admissions: '
                  '%1.4f, Benefit: %2.1f, Population size %5.0f' % (
                      generation + 1, np.max(self.score.results[:, 11]), np.max(
                          self.score.results[:, 22]),
                      self.pop.population.shape[0]))

        self.save_results()
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'End\n')

        return

    def save_results(self):
        path = self.data.output_location + '/'

        # Save results
        # Column headings for results
        cols = ['# hosp', 'mean_time_IVT', 'max_time_IVT', 'mean_time_ET',
                'max_time_ET', 'min_IVT_adm', 'max_IVT_adm', 'min_ET_adm',
                'max_ET_adm', 'prop_IVT_time', 'prop_IVT_adm', 'prop_both_IVT',
                'prop_ET_time', 'prop_ET_adm', 'prop_both_ET',
                'prop_all_targets', '95_cent_IVT', '95_cent_ET', 'transfers',
                'transfer_time', 'good_no_treat', 'good_treat',
                'add_good_per_1000', 'median_time_IVT', 'median_time_ET',
                'min_add_outcome', '5_cent_add_outcome', 
                '95_cent_add_outcome', 'max_add_outcome']

        results_df = pd.DataFrame(self.score.results, columns=cols)
        results_df.index.name = 'scenario'
        results_df.to_csv(path+'results.csv')

        # save population
        cols = list(self.data.hospitals['Hospital_name'])
        population_df = pd.DataFrame(self.pop.population, columns=cols)
        population_df.index.name = 'scenario'
        population_df.to_csv(path+'population.csv')

        first_admissions_df = pd.DataFrame(
            self.score.hospital_first_admissions, columns=cols)
        first_admissions_df.index.name = 'scenario'

        first_admissions_df.to_csv(path+'first_hospital_admissions.csv')

        hospital_thrombectomy_admissions_df = pd.DataFrame(
            self.score.hospital_thrombectomy_admissions, columns=cols)
        hospital_thrombectomy_admissions_df.index.name = 'scenario'

        hospital_thrombectomy_admissions_df.to_csv(
            path+'thrombectomy_admissions.csv')

        return


if __name__ == '__main__':
    # Initailise mode;
    model = Model()
    model.run_algorithm()
