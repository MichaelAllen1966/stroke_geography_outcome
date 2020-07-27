import datetime
import numpy as np
import os
import pandas as pd
import sys

"""
Pareto score list:

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

class Data():
    """Holds all unchanging data for model"""

    def __init__(self):
        """Initialise data"""

        # Set global variables

        ## FILE PATHS ##
        ## =============

        # Path for input data
        self.path_to_data = './data/'

        # Set directory for results output
        self.output_location = 'test'

        # Output node results for each area (usually set to false)
        self.save_node_results = False

        # Create a directory for full node results if requires
        self.output_location_node_results = self.output_location + \
                                            '/node_results/'

        ## GENERAL TRAVEL TIME CORRECTION
        # All travel times will be multiplied by this factor
        self.travel_time_correction_factor = 1.0

        ## MODEL TYPE ##
        ## =============

        # If mothership is true then all chosen units will offer thrombolysis
        # and thrombectomy. Thrombectomy may be 'disabled' by setting an arrival
        # to puncture time of 1000 min        
        self.mothership = False
        
	    # Keep all centres open (except forced closed) and
        # change balance of ET/IVT
        # All units open for IVT (except forced closed)
        # Population determines which are available for ET
        # THIS TYPE OF MODEL RUNS SLOWER !!!!
        self.vary_et_centres = False

        # Use pre-hospital diagnostic test in model?
        self.use_diagnostic = False

        ## PARETO SCORES TO USE ##
        ##  ======================

        # Which outcome results to use in Pareto selection
        self.pareto_scores_used = \
            [0,1,3,5,6]

        # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

        ## GENERAL STROKE SETTINGS ##
        ## ==========================

        # Set target travel time or distance
        self.target_travel_thrombolysis = 30
        self.target_travel_thrombectomy = 150

        # Set target for minimum sustainable admissions
        self.target_thrombolysis_admissions = 600
        self.target_thrombectomy_admissions = 150

        # Set onset to ambulance departure for hospital
    
        self.onset_to_travel = 60

        # Set hospital door to needle (thrombolysis)
        self.door_to_needle = 40

        # Set hospital door to puncture (thrombectomy)
        self.door_to_puncture = 90
        # Set net delay in time to thrombectomy (not including travel time) if
        # patient requires inter-hospital transfer
        self.transfer_net_delay = 60

        # Set allowable_delay in order to directly attend a CSC rather than HASU
        self.allowable_delay = 0

        # Set proportions of all patients eligable for treatment

        # Adjusted prop lvo eligible to give 10% all strokes have et (assume
        # preceeded by ivt). Adjustment also made that not all ET have previously
        # received IVT

        prop_ET_previously_received_IVT = 0.87
        self.prop_lvo_eligible_ivt = 0.3125 * prop_ET_previously_received_IVT
        self.prop_thrombolysed_lvo_receiving_thrombectomy = \
            1 / prop_ET_previously_received_IVT

        # Adjusted prop nlvo eligibleto give 20% of isch stroke thrombolysis
        # rate (pre ET)
        self.prop_nlvo_eligible_treatment = 0.1936

        # Stroke types (from de la Ossa et al. Stroke (2014) 45: 87-91)
        # Used when diagnostic test not applied

        self.prop_lvo = 0.3171
        self.prop_nlvo = 0.5165
        self.prop_ich = 0.1664
        self.prop_mimic = 0

        ## USE OF PREHOSPITAL DIAGNOSTIC ##
        ## ================================

        # Proportion of patients considered for testing. This is used to adjust
        # admission numbers at different units. It is assumed it does not effect
        # overall outcomes (it is assumed this portion will never be treated)

        self.proportion_tested = 1

        # Set a level of improvement (improved # good outcomes per 1,000
        # admissions required to bipass local thrombolysis unit
        #  for thrombectomy unit. Values is improvement in outcomes for 1,000
        # admissions in diagnostic group

        self.diagnostic_outcome_signifciant = 0


        # from de la Ossa et al. Stroke (2014) 45: 87-91
        # This breakdown removes mimics and RACE >= 5

        # Proportion of patients having a positive test
        self.diagnostic_prop_positive = 0.474
        self.diagnostic_prop_negative = 1 - self.diagnostic_prop_positive

        # breakdown of positive test stroke types
        self.diagnostic_pos_lvo = 0.507
        self.diagnostic_pos_nlvo = 0.263
        self.diagnostic_pos_ich = 0.230
        self.diagnostic_pos_mimic = 0

        # breakdown of negative test stroke types
        self.diagnostic_neg_lvo = 0.146
        self.diagnostic_neg_nlvo = 0.744
        self.diagnostic_neg_ich = 0.110
        self.diagnostic_neg_mimic = 0
        
        # Over-ride diagnostic performance if want to attend unit with best
        # outcome with no diagnostic
        
        assess_best_outcome_without_diagnostic = False
        
        if assess_best_outcome_without_diagnostic:
            self.diagnostic_prop_positive = 1
            # Set proportions to be equal to all strokes
            self.diagnostic_pos_lvo = self.prop_lvo
            self.diagnostic_pos_nlvo = self.prop_nlvo
            self.diagnostic_pos_ich = self.prop_ich
            self.diagnostic_pos_mimic = self.prop_mimic
                    

        ## GENETIC ALGORITHM SETTINGS ##
        ## =============================

        # Maximum number of generations in genetic algorithm
        # Set to zero to score loaded population
        self.maximum_generations = 10

        # Allow hospitals to be forced open/closed
        self.fix_hospital_status = False

        # Genetic algoruth population sizes
        self.initial_random_population_size = 5000
        self.minimum_population_size = 500
        self.maximum_population_size = 10000

        # Add new random members each generation
        self.proportion_new_random_each_generation = 0.05

        # Genetic algorithm mutation rate and crosservers
        self.mutation_rate = 0.005
        self.max_crossover_points = 3

        # Allow first generation to be scored wiiithout selection
        # If maximum generations = 1, this will simply score provided starting
        # population

        self.apply_pareto_to_starting_population = False

        ## CREATE DIRECTORY IF NEEDED AND RUN LOAD DATA ##
        ## ===============================================

        # Create output folders if neeed
        """Create new folder if folder does not already exist"""
        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)

        if self.save_node_results:
            if not os.path.exists(self.output_location_node_results):
                os.makedirs(self.output_location_node_results)

        # Load datsa
        self.load_data()
        return

    def convert_pandas_to_numpy(self):
        if not self.mothership:
            self.np_closest_thrombectomy_to_each_area_index = \
                self.closest_thrombectomy_unit_to_each_area['index_#'].values
        
            self.np_closest_thrombectomy_to_each_area_time = \
                self.closest_thrombectomy_unit_to_each_area['time'].values
        
            self.np_closest_neighbouring_thrombectomy_unit_index = \
                self.closest_neighbouring_thrombectomy_unit['index_#'].values
        
            self.np_closest_neighbouring_thrombectomy_unit_time = \
                self.closest_neighbouring_thrombectomy_unit[
                    'transfer_time'].values
            return
    
    def identify_closest_neighbouring_thrombectomy_unit(
            self, scenario=np.array([])):
        """
        For each thrombolysis unit identify closest thrombectomy unit
        """
        
        self.closest_neighbouring_thrombectomy_unit = pd.DataFrame()
        if len(scenario)>0:
            mask = scenario == 1
            mask = pd.Series(mask, index=self.hospitals.index)
        else:
            mask = self.hospitals['Fixed'] == 2


        thrombectomy_units = mask.index[mask]
        self.thrombectomy_boolean = mask 
        travel_matrix_thrombectomy = \
            self.inter_hospital_travel_matrix[thrombectomy_units]
        closest_unit = travel_matrix_thrombectomy.idxmin(axis=1)
        time_to_closest_unit = travel_matrix_thrombectomy.min(axis=1)

        # Transfer results to dataframe
        self.closest_neighbouring_thrombectomy_unit['thrombolysis_unit'] = \
            self.hospitals.index

        self.closest_neighbouring_thrombectomy_unit['thrombectomy_unit'] = \
            closest_unit.values

        self.closest_neighbouring_thrombectomy_unit['transfer_time'] = \
            time_to_closest_unit.values

        self.closest_neighbouring_thrombectomy_unit = \
            pd.merge(self.closest_neighbouring_thrombectomy_unit,
                     self.hospitals[['index_#', 'hospital']],
                     left_on='thrombectomy_unit',
                     right_on='Postcode',
                     how='left')

        return


    def identify_closest_thrombectomy_unit_to_each_patient_area(
            self, scenario=np.array([])):
        """
        For each patient area (e.g. LSAO) identify closest thrombectomy unit.
        """
        self.closest_thrombectomy_unit_to_each_area=pd.DataFrame()
        if len(scenario)>0:
            mask = scenario == 1
            mask = pd.Series(mask, index=self.hospitals.index)
        else:
            mask = self.hospitals['Fixed'] == 2
        thrombectomy_units = mask.index[mask]
        self.thrombectomy_boolean = mask
        travel_matrix_thrombectomy = self.travel_matrix[thrombectomy_units]
        closest_unit = travel_matrix_thrombectomy.idxmin(axis=1)
        time_to_closest_unit = travel_matrix_thrombectomy.min(axis=1)

        # Transfer results to dataframe
        self.closest_thrombectomy_unit_to_each_area['area'] = \
            self.travel_matrix.index
        self.closest_thrombectomy_unit_to_each_area['thrombectomy_unit'] = \
            closest_unit.values
        self.closest_thrombectomy_unit_to_each_area['time'] = \
            time_to_closest_unit.values

        self.closest_thrombectomy_unit_to_each_area = \
            pd.merge(self.closest_thrombectomy_unit_to_each_area,
                     self.hospitals[['index_#', 'hospital']],
                     left_on='thrombectomy_unit',
                     right_on='Postcode',
                     how='left')

        return


    def load_data(self):
        """load data files"""

        path = self.path_to_data

        # Define attributes
        self.admissions = []
        self.admissions_index = []
        self.hospitals = []
        self.travel_matrix = []
        self.travel_matrix_area_index = []
        self.travel_matrix_hopspital_index = []
        self.inter_hospital_travel_matrix = []
        self.closest_neighbouring_thrombectomy_unit = pd.DataFrame()
        self.closest_thrombectomy_unit_to_each_area = pd.DataFrame()
        self.hospital_count = 0
        self.loaded_population = []

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              'Loading data')
        # Load hospital list
        self.hospitals = pd.read_csv(path + 'hospitals.csv', index_col=0)
        self.hospital_count = len(self.hospitals)
        self.hospitals['index_#'] = list(range(0, self.hospital_count))
        self.hospitals['hospital'] = self.hospitals.index

        # Load admissions and split index from data
        self.admissions_with_index = pd.read_csv(path + 'admissions.csv')
        self.admissions_index = self.admissions_with_index['area']
        self.admissions = self.admissions_with_index['Admissions']

        # Load time/distance matrix
        self.travel_matrix = pd.read_csv(path + 'travel_matrix.csv',
                                         index_col=0, low_memory=False)

        self.travel_matrix *= self.travel_time_correction_factor

        # Load inter-hospital travel time matrix
        self.inter_hospital_travel_matrix = \
            pd.read_csv(path + 'inter_hospital_time.csv', index_col=0)

        self.inter_hospital_travel_matrix *= self.travel_time_correction_factor

        # Load initial population if data/load.csv exists
        try:
            self.loaded_population = np.loadtxt(path + 'load.csv',
                                                delimiter=',')

            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                  'Loaded starting population from file')

        except:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                  'No initial population loaded from file')

        # Identify closest thrombectomy unit to each hospital (IF NOT MOTHERSHIP
        # MODEL)

        if not self.mothership:
            self.identify_closest_neighbouring_thrombectomy_unit()

            # Identify closest thrombectomy unit to each patient area
            # (e.g. LSOA)
            self.identify_closest_thrombectomy_unit_to_each_patient_area()

        # Set up NumPy arrays for faster work

        self.np_admissions = self.admissions.values

        self.np_travel_matrix = self.travel_matrix.values
        
        self.convert_pandas_to_numpy()


        return
