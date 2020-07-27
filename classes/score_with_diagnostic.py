##TO DO Consider adding in 'not suitable for test'

"""
Score population according to:
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


import numpy as np
import pandas as pd
from classes.clinical_outcome import Clinical_outcome


class Score_population_with_diagnostic():
    def __init__(self, data, population):
        number_of_scenarios = population.shape[0]
        number_of_hospitals = population.shape[1]
        number_of_areas = len(data.np_admissions)
        total_admissions = sum(data.admissions)

        # Set up results tables
        self.results = np.zeros((number_of_scenarios, 29))
        self.hospital_first_admissions = np.zeros((population.shape))
        self.hospital_thrombectomy_admissions = np.zeros((population.shape))
        node_results = np.zeros((number_of_areas, 47))

        # Set up clinical outcome object
        self.outcome = Clinical_outcome()

        """
        Node results are results for each area (e.g. LSAO)
        
        # General measures
        
         0: Time to closest hospital
         1: Orginal (full hosital list) index # of closest hospital
         2: Time to closest CSC (direct)
         3: Orginal (full hosital list) index # of closest CSC (direct)
         4: Transfer time to closest CSC (drip and ship) 
         5: Orginal (full hosital list) index # of closest CSC (drip and ship)
         6: Total drip and ship time: orginal transfer + net delay + transfer
         
        # Negative diagnostic test (assume go to closest)
        
         7: Negative test admissions
         8: Chosen thrombolysis centre
         9: Time to chosen thrombolysis centre
        10: Chosen thrombectomy centre
        11: Time to chosen thrombectomy centre
        12: Number of transfers to CSC
        13: Distance of transfers to CSC
        14: Clinical benefit - no treatement
        15: Additional clinical benefit
        
        # Positive diagnostic test
        
        16: Positive test admissions
        17: Clinical benefit - no treatment
        18: Additional clinical direct to CSC
        19: Additional clinical drip and ship
        20: Choose CSC
        21: Chosen thrombolysis centre
        22: Time to chosemn thrombolysis centre
        23: Chosen thrombectomy centre
        24: Time to chosen thrombectomy centre
        25: Number of transfers to CSC
        26: Distance of transfers to CSC
        27: Clinical benefit from chosen location
        
        # Adjusted admissions (takes into account people where no action woiuld
        # be taken even with positive LVO diagnostic test)
        
        28: Adjusted IVT admissions
        29: Adjusted ET admissions
                
        # Admission numbers
        
        30: -ve test thrombolysis unit admissions
        31: -ve test thrombectomy unit procedures
        32: +ve test thrombolysis unit admissions
        33: +ve test thrombectomy unit procedures
        
        # Targets met
        
        34: -ve test thrombolysis unit target admissions
        35: -ve test thrombolysis target time
        36: -ve test thrombolysis both targets
        37: -ve test thrombectomy unit target admissions
        38: -ve test thrombectomy target time
        39: -ve test thrombectomy both targets
        40: +ve test thrombolysis unit target admissions
        41: +ve test thrombolysis target time
        42: +ve test thrombolysis both targets
        43: +ve test thrombectomy unit target admissions
        44: +ve test thrombectomy target time
        45: +ve test thrombectomy both targets
        
        # Net clinical benefit
        46: Net clinical benefit
        
        """

        for i in range(number_of_scenarios):
            # Create and apply mask to remove unused hospitals in scenario
            if data.vary_et_centres:
                # Have all hospitals open for IVT except forced closed ones
                mask = data.hospitals['Fixed'] != -1
                # Recalculate travel times to ET units
                data.identify_closest_neighbouring_thrombectomy_unit(
                        population[i,:])
                data.identify_closest_thrombectomy_unit_to_each_patient_area(
                        population[i,:])
                data.convert_pandas_to_numpy()
            else:
                mask = population[i, :] == 1

            _ = data.hospitals['hospital'].values
            used_hospital_postcodes = _[mask]

            _ = data.hospitals['index_#'].values
            used_hospital_index_nos = _[mask]

            used_travel_matrix = data.np_travel_matrix[:, mask]

            # Node result 0: Identify time closest hospital
            node_results[:, 0] = np.min(used_travel_matrix, axis=1)

            # Node result 1: Identify orginal (full hosital list) index # of
            # closest hospital
            local_id = np.argmin(used_travel_matrix, axis=1)
            node_results[:, 1] = used_hospital_index_nos[local_id]

            # Node result 2: Time to closest CSC (direct)
            node_results[:, 2] = \
                data.np_closest_thrombectomy_to_each_area_time

            # Node result 3: orginal (full hosital list) index # of closest
            # CSC (direct)
            node_results[:, 3] = \
                data.np_closest_thrombectomy_to_each_area_index

            # Node result 4 & 5: Transfer time and index (original) to
            # closest CSC (drip'n'ship)
            fancy_index = np.int_(node_results[:, 1])

            node_results[:, 4] = \
                data.np_closest_neighbouring_thrombectomy_unit_time[
                    fancy_index]

            node_results[:, 5] = \
                data.np_closest_neighbouring_thrombectomy_unit_index[
                    fancy_index]

            # Node 6 Total drip and ship time (original travel + net delay +
            # transfer)
            node_results[:, 6] = (
                    node_results[:, 0] +
                    node_results[:, 4])

            # Transfer delay if thrombectomy and thrombolysis centres are different
            mask = node_results[:, 1] != node_results[:, 3]
            node_results[mask, 6] += data.transfer_net_delay

            ## NEGATIVE DIAGNOSTIC TEST RESULTS

            # Admissions with negative diagnostic test
            node_results[:, 7] = data.admissions * \
                                 (data.diagnostic_prop_negative)

            # Create mask for direct to CSC
            mask = node_results[:, 2] <= node_results[:, 0] + \
                   data.allowable_delay

            # Chosen IVT unit
            node_results[:, 8] = node_results[:, 1]
            node_results[mask, 8] = node_results[mask, 3]

            # IVT time
            node_results[:, 9] = node_results[:, 0]
            node_results[mask, 9] = node_results[mask, 2]

            # Chosen ET unit
            node_results[:, 10] = node_results[:, 5]
            node_results[mask, 10] = node_results[mask, 3]

            # ET time
            node_results[:, 11] = node_results[:, 6]
            node_results[mask, 11] = node_results[mask, 2]

            # Number of transfers for drip and ship
            node_results[:, 12] = \
                (node_results[:, 7] *
                 data.diagnostic_neg_lvo *
                 data.prop_lvo_eligible_ivt *
                 data.prop_thrombolysed_lvo_receiving_thrombectomy)

            node_results[mask, 12] = 0

            # Distance of transfers for drip and ship
            node_results[:, 13] = node_results[:, 12] * node_results[:, 4]

            # Clinical benefit for negative diagnostic test

            admissions = data.admissions.values * data.diagnostic_prop_negative
            areas = len(admissions)

            mimic = np.ones(areas) * data.diagnostic_neg_mimic
            ich = np.ones(areas) * data.diagnostic_neg_ich
            nlvo = np.ones(areas) * data.diagnostic_neg_nlvo
            lvo = np.ones(areas) * data.diagnostic_neg_lvo

            prop_nlvo_eligible_treatment = np.zeros(len(admissions))
            prop_nlvo_eligible_treatment.fill(data.prop_nlvo_eligible_treatment)

            prop_lvo_eligible_treatment = np.zeros(len(admissions))
            prop_lvo_eligible_treatment.fill(data.prop_lvo_eligible_ivt)

            door_to_needle = data.door_to_needle


            onset_to_needle = (data.onset_to_travel +
                               door_to_needle +
                               node_results[:, 0])

            onset_to_puncture = (data.onset_to_travel +
                                 data.door_to_puncture +
                                 node_results[:, 11])

            # Get outcome with no treatment

            no_treatment_outcome = (
                    data.diagnostic_neg_lvo * 0.1328 +
                    data.diagnostic_neg_nlvo * 0.4622 +
                    data.diagnostic_neg_ich * 0.24 +
                    data.diagnostic_neg_mimic * 1)

            node_results[:, 14] = np.ones(areas) * no_treatment_outcome

            # Get outcome with treatment
            outcome = self.outcome.calculate_outcome_for_all(
                mimic,
                ich,
                nlvo,
                lvo,
                onset_to_needle,
                onset_to_puncture,
                prop_nlvo_eligible_treatment,
                prop_lvo_eligible_treatment,
                data.prop_thrombolysed_lvo_receiving_thrombectomy)

            # Calculate additional clinical benefit from treatment
            node_results[:, 15] = outcome - node_results[:, 14]

            ## POSITIVE DIAGNISTIC TEST RESULTS

            # To choose between direct to CSC or drip and ship for each area,
            # compare clinical outcomes, and choose the best clinical outcome

            # Record admissions for positive test

            admissions = data.admissions.values * data.diagnostic_prop_positive

            node_results[:, 16] = admissions

            # Clinical benefit direct to CSC
            door_to_needle = data.door_to_needle

            onset_to_needle = (data.onset_to_travel +
                               door_to_needle +
                               node_results[:, 2])

            onset_to_puncture = \
                data.onset_to_travel + data.door_to_puncture + \
                node_results[:, 2]

            # Get outcome with no treatment

            no_treatment_outcome = (
                    data.diagnostic_pos_lvo * 0.1328 +
                    data.diagnostic_pos_nlvo * 0.4622 +
                    data.diagnostic_pos_ich * 0.24 +
                    data.diagnostic_pos_mimic * 1)

            node_results[:, 17] = no_treatment_outcome

            # Get clinical benefit with treatment

            mimic = np.ones(areas) * data.diagnostic_pos_mimic
            ich = np.ones(areas) * data.diagnostic_pos_ich
            nlvo = np.ones(areas) * data.diagnostic_pos_nlvo
            lvo = np.ones(areas) * data.diagnostic_pos_lvo

            outcome = self.outcome.calculate_outcome_for_all(
                mimic,
                ich,
                nlvo,
                lvo,
                onset_to_needle,
                onset_to_puncture,
                prop_nlvo_eligible_treatment,
                prop_lvo_eligible_treatment,
                data.prop_thrombolysed_lvo_receiving_thrombectomy)

            # Calculate added benefit with direct to thrombectomy centre
            node_results[:, 18] = outcome - node_results[:, 17]

            # Clinical benefit drip and ship
            door_to_needle = data.door_to_needle
            
            onset_to_needle = (data.onset_to_travel +
                               door_to_needle +
                               node_results[:, 9])

            onset_to_puncture = (data.onset_to_travel +
                                 data.door_to_puncture +
                                 node_results[:, 6])

            outcome = self.outcome.calculate_outcome_for_all(
                mimic,
                ich,
                nlvo,
                lvo,
                onset_to_needle,
                onset_to_puncture,
                prop_nlvo_eligible_treatment,
                prop_lvo_eligible_treatment,
                data.prop_thrombolysed_lvo_receiving_thrombectomy)

            node_results[:, 19] = outcome - node_results[:, 17]

            # Create mask for direct to CSC
            # debug make drip and ship for everyone
            mask = node_results[:, 18] >= node_results[:, 19] + \
                   (data.diagnostic_outcome_signifciant/1000)

            # Record direct to CSC (convert Boolean to 0/1)
            node_results[:, 20] = mask * 1

            # Chosen IVT unit
            node_results[:, 21] = node_results[:, 1]
            node_results[mask, 21] = node_results[mask, 3]

            # IVT time
            node_results[:, 22] = node_results[:, 0]
            node_results[mask, 22] = node_results[mask, 2]

            # Chosen ET unit
            node_results[:, 23] = node_results[:, 5]
            node_results[mask, 23] = node_results[mask, 3]

            # ET time
            node_results[:, 24] = node_results[:, 6]
            node_results[mask, 24] = node_results[mask, 2]

            # Number of transfers for drip and ship
            node_results[:, 25] = \
                (node_results[:, 16] *
                 data.diagnostic_pos_lvo *
                 data.prop_lvo_eligible_ivt *
                 data.prop_thrombolysed_lvo_receiving_thrombectomy)

            node_results[mask, 25] = 0

            # Distance of transfers for drip and ship
            node_results[:, 26] = node_results[:, 25] * node_results[:, 4]

            # Clinical benefit of chosen hospital
            node_results[:, 27] = node_results[:, 19]
            node_results[mask, 27] = node_results[mask, 18]

            # Adjusted admissions
            # IVT admitting unit includes positive diagnostic test fraction
            # where no diversion would take place (e.g. certain outside
            # window. ET admitting unit reduced by the same number

            #  ADMISSION NUMBERS

            # Adjusted IVT admissions (includes 'no action on test' patients)
            node_results[:, 28] = \
                node_results[:, 7] + \
                node_results[:, 16] * (1 - data.proportion_tested)

            # Adjust ET admissions (reduced by 'no action on test' patients)
            node_results[:, 29] = node_results[:, 16] * data.proportion_tested

            # Non-adjusted admissions are used to calculate total thrombectomies
            non_adjusted_admissions_concatenated = np.concatenate(
                (node_results[:, 7], node_results[:, 16]))

            # Adjusted admissions are used to calculate first admitting hospital
            adjusted_admissions_concatenated = np.concatenate(
                (node_results[:, 28], node_results[:, 29]))

            admitting_ivt_hospital = np.concatenate((node_results[:, 8],
                                                     node_results[:, 21]))

            admitting_et_hospital = np.concatenate((node_results[:, 10],
                                                    node_results[:, 23]))

            thrombolysis_admissions_by_hospital = np.bincount(
                np.int_(admitting_ivt_hospital),
                weights=adjusted_admissions_concatenated)

            thrombectomy_admissions_by_hospital = np.bincount(
                np.int_(admitting_et_hospital),
                weights=non_adjusted_admissions_concatenated)

            overall_proportion_of_lvo_eligible_for_treatment = (
                    (data.diagnostic_prop_positive *
                     data.diagnostic_pos_lvo *
                     data.prop_lvo_eligible_ivt) +
                    ((data.diagnostic_prop_negative) *
                     data.diagnostic_neg_lvo *
                     data.prop_lvo_eligible_ivt))

            thrombectomy_admissions_by_hospital *= \
                (overall_proportion_of_lvo_eligible_for_treatment *
                 data.prop_thrombolysed_lvo_receiving_thrombectomy)

            # Fill in missing hospital counts at end of array
            if len(thrombolysis_admissions_by_hospital) < number_of_hospitals:
                zeros_to_add = number_of_hospitals - \
                               len(thrombolysis_admissions_by_hospital)

                thrombolysis_admissions_by_hospital = \
                    np.hstack((thrombolysis_admissions_by_hospital,
                               np.zeros(zeros_to_add)))

            if len(thrombectomy_admissions_by_hospital) < number_of_hospitals:
                zeros_to_add = number_of_hospitals - \
                               len(thrombectomy_admissions_by_hospital)

                thrombectomy_admissions_by_hospital = \
                    np.hstack((thrombectomy_admissions_by_hospital,
                               np.zeros(zeros_to_add)))

            # Record admission results
            self.hospital_first_admissions[i, :] = \
                thrombolysis_admissions_by_hospital

            self.hospital_thrombectomy_admissions[i, :] = \
                thrombectomy_admissions_by_hospital

            # Add in unit admission numbers to node results

            # -ve test thrombolysis unit admissions
            node_results[:, 30] = \
                thrombolysis_admissions_by_hospital[np.int_ \
                    (node_results[:, 8])]

            # -ve test thrombectomy unit procedures
            node_results[:, 31] = \
                (thrombectomy_admissions_by_hospital \
                    [np.int_(node_results[:, 10])])

            # +ve test thrombolysis unit admissions
            node_results[:, 32] = \
                thrombolysis_admissions_by_hospital[np.int_ \
                    (node_results[:, 21])]

            # +ve test thrombectomy unit procedures
            node_results[:, 33] = \
                (thrombectomy_admissions_by_hospital \
                    [np.int_(node_results[:, 23])])

            # RECORD TARGETS MET

            # -ve test thrombolysis unit target admissions
            node_results[:, 34] = \
                node_results[:, 30] >= data.target_thrombolysis_admissions

            # -ve test thrombolysis target time
            node_results[:, 35] = \
                node_results[:, 9] <= data.target_travel_thrombolysis

            # -ve test thrombolysis both targets
            node_results[:, 36] = \
                np.logical_and(node_results[:, 34], node_results[:, 35])

            # -ve test thrombectomy unit target admissions
            node_results[:, 37] = \
                node_results[:, 31] >= data.target_thrombectomy_admissions

            # -ve test thrombectomy target time
            node_results[:, 38] = \
                node_results[:, 11] <= data.target_travel_thrombectomy

            # -ve test thrombectomy both targets
            node_results[:, 39] = \
                np.logical_and(node_results[:, 37], node_results[:, 38])

            # +ve test thrombolysis unit target admissions
            node_results[:, 40] = \
                node_results[:, 32] >= data.target_thrombolysis_admissions

            # +ve test thrombolysis target time
            node_results[:, 41] = \
                node_results[:, 22] <= data.target_travel_thrombolysis

            # +ve test thrombolysis both targets
            node_results[:, 42] = \
                np.logical_and(node_results[:, 40], node_results[:, 41])

            # +ve test thrombectomy unit target admissions
            node_results[:, 43] = \
                node_results[:, 33] >= data.target_thrombectomy_admissions

            # +ve test thrombectomy target time
            node_results[:, 44] = \
                node_results[:, 24] <= data.target_travel_thrombectomy

            # +ve test thrombectomy both targets
            node_results[:, 45] = \
                np.logical_and(node_results[:, 43], node_results[:, 44])

            # Net clinical benefit (weighted benefit by diagnostic test
            # proportion)
            node_results[:, 46] = (
                    node_results[:, 15] * data.diagnostic_prop_negative +
                    node_results[:, 27] * data.diagnostic_prop_positive)

            # Save full node results (not usually used)

            if data.save_node_results:
                filename = './' + data.output_location_node_results + \
                           str(i) + '.csv'
                node_df = pd.DataFrame()
                node_df['area'] = data.admissions_index.values
                node_df['admissions'] = data.admissions.values

                # Add negative test reults
                node_df['neg_test_admissions_inc_no_test'] = node_results[:, 28]
                node_df['neg_test_IVT_unit_#'] = node_results[:, 8]
                node_df['neg_test_time_to_IVT_unit'] = node_results[:, 9]
                node_df['neg_test_ET_unit_#'] = node_results[:, 10]
                node_df['neg_test_time_to_ET_unit'] = node_results[:, 11]
                node_df['neg_add_clinc_benefit'] = node_results[:, 15] * 1000

                # Add IVT hospital names
                node_df = pd.merge(node_df,
                                   data.hospitals[['index_#', 'Hospital_name']],
                                   left_on='neg_test_IVT_unit_#',
                                   right_on='index_#',
                                   how='left')

                # Delete unecessary columns
                node_df.drop(
                    ['index_#', 'neg_test_IVT_unit_#'], axis=1, inplace=True)

                # Rename hospital name column
                node_df = node_df.rename \
                    (columns={'Hospital_name': 'neg_test_IVT unit'})

                # Add ET hospital names
                node_df = pd.merge(node_df,
                                   data.hospitals[['index_#', 'Hospital_name']],
                                   left_on='neg_test_ET_unit_#',
                                   right_on='index_#',
                                   how='left')

                # Delete unecessary columns
                node_df.drop(
                    ['index_#', 'neg_test_ET_unit_#'], axis=1, inplace=True)

                # Rename hospital name column
                node_df = node_df.rename \
                    (columns={'Hospital_name': 'neg_test_ET unit'})

                # Add positive test reults
                node_df['pos_test_admissions_inc_no_test'] = node_results[:, 29]
                node_df['pos_test_IVT_unit_#'] = node_results[:, 21]
                node_df['pos_test_time_to_IVT_unit'] = node_results[:, 22]
                node_df['pos_test_ET_unit_#'] = node_results[:, 23]
                node_df['pos_test_time_to_ET_unit'] = node_results[:, 24]
                node_df['pos_add_clinc_benefit'] = node_results[:, 27] * 1000
                node_df['pos_add_clinc_benefit_direct_to_CSC'] = \
                    (node_results[:, 18] - node_results[:, 19]) * 1000

                # Add IVT hospital names
                node_df = pd.merge(node_df,
                                   data.hospitals[['index_#', 'Hospital_name']],
                                   left_on='pos_test_IVT_unit_#',
                                   right_on='index_#',
                                   how='left')

                # Delete unecessary columns
                node_df.drop(
                    ['index_#', 'pos_test_IVT_unit_#'], axis=1, inplace=True)

                # Rename hospital name column
                node_df = node_df.rename \
                    (columns={'Hospital_name': 'pos_test_IVT unit'})

                # Add ET hospital names
                node_df = pd.merge(node_df,
                                   data.hospitals[['index_#', 'Hospital_name']],
                                   left_on='pos_test_ET_unit_#',
                                   right_on='index_#',
                                   how='left')

                # Delete unecessary columns
                node_df.drop(
                    ['index_#', 'pos_test_ET_unit_#'], axis=1, inplace=True)

                # Rename hospital name column
                node_df = node_df.rename \
                    (columns={'Hospital_name': 'pos_test_ET unit'})

                # Add net clinical benefit
                node_df['net_clinical_benefit_per_1000'] = \
                    node_results[:, 46] * 1000

                # save results
                node_df.index.name = 'scenario'
                node_df.to_csv(filename)

            # COLLATE SUMMARY RESULTS

            # Result 0: Number of hospitals
            self.results[i, 0] = len(used_hospital_postcodes)

            # Time to thrombolysis results (odd sequence due to original order
            # aded to results)

            # Result 1: mean time to thrombolysis
            travel_time = np.concatenate((node_results[:, 9],
                                          node_results[:, 22]))

            patients = np.concatenate((node_results[:, 28],
                                       node_results[:, 29]))

            self.results[i, 1] = \
                np.sum(travel_time * patients) / np.sum(patients)

            # Result 2: Maximum time to thrombolysis
            # Use mask where admissions >0
            mask = data.admissions > 0
            mask2 =  np.concatenate((mask,mask))

            self.results[i, 2] = np.max(travel_time[mask2])

            # Result 23: Median time to thrombolysis
            self.results[i, 23] = self.calculate_weighted_percentiles(
                travel_time, patients, [0.5])[0]

            # Result 16: 95th pecentile time to thrombolysis
            self.results[i, 16] = self.calculate_weighted_percentiles(
                travel_time, patients, [0.95])[0]

            # Time to thrombectomy results

            travel_time = np.concatenate((node_results[:, 11],
                                          node_results[:, 24]))

            # Result 3: Mean time to thrombectomy

            self.results[i, 3] = \
                np.sum(travel_time * patients) / np.sum(patients)

            # Result 4: Maximum time to thrombectomy
            self.results[i, 4] = np.max(travel_time[mask2])

            # Result 23: Median time to thrombolysis
            self.results[i, 24] = self.calculate_weighted_percentiles(
                travel_time, patients, [0.5])[0]

            # Result 16: 95th pecentile time to thrombolysis
            self.results[i, 17] = self.calculate_weighted_percentiles(
                travel_time, patients, [0.95])[0]

            # Result 5 & 6: Minimum and maximum thrombolysis admissions to any
            # one hospital
            if data.vary_et_centres:
                # Have all hospitals open for IVT except forced closed ones
                mask = data.hospitals['Fixed'] != -1
            else:
                mask = population[i, :] == 1
            admissions_to_used_units = thrombolysis_admissions_by_hospital[mask]
            self.results[i, 5] = np.min(admissions_to_used_units)
            self.results[i, 6] = np.max(admissions_to_used_units)

            # Result 7 & 8: Minimum and maximum thrombectomy admissions to any
            # one hospital.

            mask = data.thrombectomy_boolean

            admissions_to_used_units = thrombectomy_admissions_by_hospital[mask]

            self.results[i, 7] = np.min(admissions_to_used_units)
            self.results[i, 8] = np.max(admissions_to_used_units)

            # Result 9: Proportion patients within target thrombolysis time

            target_met = np.concatenate(
                (node_results[:, 35], node_results[:, 41]))

            self.results[i, 9] = (np.sum(patients * target_met) /
                                  np.sum(patients))

            # Result 10: Proportion patients attending unit with target first
            # admissions

            target_met = np.concatenate(
                (node_results[:, 34], node_results[:, 40]))

            self.results[i, 10] = (np.sum(patients * target_met) /
                                   np.sum(patients))

            # Result 11: Proportion patients meeting both thrombolysis targets

            target_met = np.concatenate(
                (node_results[:, 36], node_results[:, 42]))

            self.results[i, 11] = (np.sum(patients * target_met) /
                                   np.sum(patients))

            # Result 12: Proportion patients within target thrombectomy time

            target_met = np.concatenate(
                (node_results[:, 38], node_results[:, 44]))

            self.results[i, 12] = (np.sum(patients * target_met) /
                                   np.sum(patients))

            # Result 13: Proportion patients attending unit with target
            # thrombectomy admissions

            target_met = np.concatenate(
                (node_results[:, 37], node_results[:, 43]))

            self.results[i, 13] = (np.sum(patients * target_met) /
                                   np.sum(patients))

            # Result 14: Proportion patients meeting both thrombectomy targets

            target_met = np.concatenate(
                (node_results[:, 39], node_results[:, 45]))

            self.results[i, 14] = (np.sum(patients * target_met) /
                                   np.sum(patients))

            # Result 15: Proportion patients meeting all targets

            areas_meeting_all_targets = \
                np.min(node_results[:, [36, 39, 42, 45]], axis=1)

            self.results[i, 15] = \
                np.sum(areas_meeting_all_targets * admissions) / \
                np.sum(admissions)

            # Result 18: Total transfers
            self.results[i, 18] = np.sum(node_results[:, 12])

            # Result 19: Total transfer time
            self.results[i, 19] = np.sum(node_results[:, 13])

            # Result 20-22: Net clinical outcome

            # Note clinical outcome is based on patients  in -ve and +ve
            # diagnostic test groups before removal on non-tested patients

            # Result 20 No treatment
            patients = np.concatenate((node_results[:, 7], node_results[:, 16]))
            outcome = np.concatenate((node_results[:, 14], node_results[:, 17]))
            self.results[i, 20] = np.sum(outcome * patients) / np.sum(patients)
            self.results[i, 20] *= 1000

            # Result 22 average added good outcomes per 1,000 patients

            outcome = np.concatenate((node_results[:, 15], node_results[:, 27]))
            self.results[i, 22] = np.sum(outcome * patients) / np.sum(patients)

            # Express outcome as good outcomes per 1000 patients
            self.results[i, 22] *= 1000

            # Total good oucomes with treatment
            self.results[i, 21] = self.results[i, 20] + self.results[i, 22]
            
            # Results 25-28 added clinical outcome ranges
            
            self.results[i, 25] = np.min(node_results[:, 46]) * 1000

            self.results[i, 26] = self.calculate_weighted_percentiles(
                node_results[:, 46], data.admissions.values, [0.05])[0] * 1000
            
            self.results[i, 27] = self.calculate_weighted_percentiles(
                node_results[:, 46], data.admissions.values, [0.95])[0] * 1000
            
            self.results[i, 28] = np.max(node_results[:, 46]) * 1000


        return

    @staticmethod
    def calculate_weighted_percentiles(data, wt, percentiles):
        """Calculate weighted percentiles. Multiple percentiles may be passed as
         a list"""

        assert np.greater_equal(percentiles,
                                0.0).all(), "Percentiles less than zero"

        assert np.less_equal(percentiles,
                             1.0).all(), "Percentiles greater than one"

        data = np.asarray(data)

        assert len(data.shape) == 1

        if wt is None:
            wt = np.ones(data.shape, np.float)
        else:
            wt = np.asarray(wt, np.float)
            assert wt.shape == data.shape
            assert np.greater_equal(wt, 0.0).all(), "Not all weights are " \
                                                    "non-negative."
        assert len(wt.shape) == 1
        n = data.shape[0]
        assert n > 0
        i = np.argsort(data)
        sd = np.take(data, i, axis=0)
        sw = np.take(wt, i, axis=0)
        aw = np.add.accumulate(sw)
        if not aw[-1] > 0:
            raise ValueError('Nonpositive weight sum')
        w = (aw - 0.5 * sw) / aw[-1]
        spots = np.searchsorted(w, percentiles)
        o = []
        for (s, p) in zip(spots, percentiles):
            if s == 0:
                o.append(sd[0])
            elif s == n:
                o.append(sd[n - 1])
            else:
                f1 = (w[s] - p) / (w[s] - w[s - 1])
                f2 = (p - w[s - 1]) / (w[s] - w[s - 1])
                assert f1 >= 0 and f2 >= 0 and f1 <= 1 and f2 <= 1
                assert abs(f1 + f2 - 1.0) < 1e-6
                o.append(sd[s - 1] * f1 + sd[s] * f2)
        return o
