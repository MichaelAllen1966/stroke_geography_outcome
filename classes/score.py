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


class Score_population():
    def __init__(self, data, population):
        number_of_scenarios = population.shape[0]
        number_of_hospitals = population.shape[1]
        number_of_areas = len(data.np_admissions)
        total_admissions = sum(data.admissions)

        # Set up results tables
        self.results = np.zeros((number_of_scenarios, 29))
        self.hospital_first_admissions = np.zeros((population.shape))
        self.hospital_thrombectomy_admissions = np.zeros((population.shape))
        node_results = np.zeros((number_of_areas, 25))

        # Set up clinical outcome object
        self.outcome = Clinical_outcome()

        """
        Node results are results for each area (e.g. LSAO)
        
         0: Time to closest hospital
         1: Orginal (full hosital list) index # of closest hospital
         2: Time to closest CSC (direct)
         3: Orginal (full hosital list) index # of closest CSC (direct)
         4: Transfer time to closest CSC (drip and ship) 
         5: Orginal (full hosital list) index # of closest CSC (drip and ship)
         6: Total drip and ship time: orginal transfer + net delay + transfer
         7: Time to chosen first (thrombolysis) hospital
         8: Orginal (full hosital list) index # of chosen first hospital 
         9: Time to chosen CSC (thrombectomy)
        10: Orginal (full hosital list) index # of chosen (thrombectomy)
        11: # of drip and ship Transfers to CSC (one way)
        12: Total time of drip and ship Transfers to CSC (one way)
        13: Thrombolysis admissions to hospital attended
        14: Thrombectomy admissions to hopspital attended for thrombectomy
        15: Thrombolysis within target time
        16: Thrombolysis at hospital with target admissions
        17: Thrombolysis meeting time and admissions targets 
        18: Thrombectomy within target time
        19: Thrombectomy at hospital with target admissions
        20: Thrombectomy meeting time and admissions targets
        21: All thrombolysis * thrombectomy targets met
        22: Base clinical outcome (per 1,000 patients)
        23: Tretaed clinical outcome (per 1,000 patients)
        24: Additional clinical outcome (per 1,000 patients)     
        """
        # Set up variables used in clinical outcome call
        admissions = data.admissions.values
        areas = len(admissions)

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

            # Do not set the following for mothership model
            if not data.mothership:
                
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

                # Create mask for direct to CSC

                mask = node_results[:, 2] <= node_results[:, 0] + \
                       data.allowable_delay

                # Fill in info for chosen hospitals for first admission and CSC
                # (thrombectomy). First fill in for all assuming drip'n'ship,
                # and then overwrite those direct to CSC

                # Nodes 7 and 8: time to, and index of, first hospital for
                # thrombolysis
                node_results[:, 7] = node_results[:, 0]
                node_results[mask, 7] = node_results[mask, 2]

                node_results[:, 8] = node_results[:, 1]
                node_results[mask, 8] = node_results[mask, 3]

                # Nodes 9 & 10: time to, and index of, hospital for thrombectomy
                node_results[:, 9] = node_results[:, 6]
                node_results[mask, 9] = node_results[mask, 2]

                node_results[:, 10] = node_results[:, 5]
                node_results[mask, 10] = node_results[mask, 3]

                # Nodes 11 & 12 Number and total distance (transfer * patients)
                # for drip and ship

                node_results[:, 11] = \
                    (data.admissions *
                     data.prop_lvo_eligible_ivt *
                     data.prop_lvo *
                     data.prop_thrombolysed_lvo_receiving_thrombectomy)

                node_results[mask, 11] = 0

                node_results[:, 12] = node_results[:, 11] * \
                                      node_results[:, 4]
                node_results[mask, 12] = 0


            else:
                # Mothership: tthrombolysis/thrombectomy distance and unit is
                # same as thrombolysis
                node_results[:, 3] = node_results[:, 0]
                node_results[:, 4] = np.zeros(len(admissions))
                node_results[:, 5] = node_results[:, 1]
                node_results[:, 6] = node_results[:, 0]
                node_results[:, 7] = node_results[:, 0]
                node_results[:, 8] = node_results[:, 1]
                node_results[:, 9] = node_results[:, 0]
                node_results[:, 10] = node_results[:, 1]
                node_results[:, 11] = np.zeros(len(admissions))
                node_results[:, 12] = np.zeros(len(admissions))

            # Get admission numbers for hospitals. Note that hospital ids
            # greater than the highest used hospital are not included in
            # bincount. If bincount has lower length than hospital count, zeroes
            # are added

            thrombolysis_admissions_by_hospital = np.bincount(
                np.int_(node_results[:, 8]), weights=data.admissions)

            # Fill in missing hospital counts at end of array
            if len(thrombolysis_admissions_by_hospital) < number_of_hospitals:
                zeros_to_add = number_of_hospitals - \
                               len(thrombolysis_admissions_by_hospital)

                thrombolysis_admissions_by_hospital = \
                    np.hstack((thrombolysis_admissions_by_hospital,
                               np.zeros(zeros_to_add)))

            self.hospital_first_admissions[i, :] = \
                thrombolysis_admissions_by_hospital

            node_results[:, 13] = \
                thrombolysis_admissions_by_hospital[np.int_(node_results[:, 8])]

            thrombectomy_admissions_by_hospital = np.bincount(
                np.int_(node_results[:, 10]), weights=data.admissions)

            thrombectomy_admissions_by_hospital *= \
                (data.prop_lvo * data.prop_lvo_eligible_ivt *
                 data.prop_thrombolysed_lvo_receiving_thrombectomy)

            # Fill in missing hospital counts at end of array
            if len(thrombectomy_admissions_by_hospital) < number_of_hospitals:
                zeros_to_add = number_of_hospitals - \
                               len(thrombectomy_admissions_by_hospital)

                thrombectomy_admissions_by_hospital = \
                    np.hstack((thrombectomy_admissions_by_hospital,
                               np.zeros(zeros_to_add)))

            self.hospital_thrombectomy_admissions[i, :] = \
                thrombectomy_admissions_by_hospital

            node_results[:, 14] = thrombectomy_admissions_by_hospital[
                np.int_(node_results[:, 10])]

            # Check whether area (LSOA) meets thrombolysis time and admissions
            # target
            node_results[:, 15] = node_results[:, 7] <= \
                                  data.target_travel_thrombolysis

            node_results[:, 16] = node_results[:, 13] >= \
                                  data.target_thrombolysis_admissions

            node_results[:, 17] = np.logical_and(
                node_results[:, 15], node_results[:, 16])

            # Check whether area (LSOA) meets thrombectomy time and admissions
            # target

            node_results[:, 18] = node_results[:, 9] <= \
                                  data.target_travel_thrombectomy

            node_results[:, 19] = node_results[:, 14] >= \
                                  data.target_thrombectomy_admissions

            node_results[:, 20] = np.logical_and(node_results[:, 18],
                                                 node_results[:, 19])

            # Check whether all taegts met
            node_results[:, 21] = np.logical_and(node_results[:, 17],
                                                 node_results[:, 20])

            # CLINICAL OUTCOME

            mimic = np.ones(areas) * data.prop_mimic
            ich = np.ones(areas) * data.prop_ich
            nlvo = np.ones(areas) * data.prop_nlvo
            lvo = np.ones(areas) * data.prop_lvo
            prop_nlvo_eligible_treatment = np.zeros(len(admissions))
            prop_nlvo_eligible_treatment.fill(data.prop_nlvo_eligible_treatment)
            prop_lvo_eligible_treatment = np.zeros(len(admissions))
            prop_lvo_eligible_treatment.fill(data.prop_lvo_eligible_ivt)

            door_to_needle = data.door_to_needle

            onset_to_needle = (data.onset_to_travel +
                               door_to_needle +
                               node_results[:, 7])

            onset_to_puncture = (data.onset_to_travel +
                                 data.door_to_puncture +
                                 node_results[:, 9])

            base_outcome = (data.prop_lvo * 0.1328 +
                            data.prop_nlvo * 0.4622 +
                            data.prop_ich * 0.24 +
                            data.prop_mimic * 1)

            node_results[:, 22] = base_outcome * 1000

            treated_outcome = self.outcome.calculate_outcome_for_all(
                mimic,
                ich,
                nlvo,
                lvo,
                onset_to_needle,
                onset_to_puncture,
                prop_nlvo_eligible_treatment,
                prop_lvo_eligible_treatment,
                data.prop_thrombolysed_lvo_receiving_thrombectomy)

            node_results[:, 23] = treated_outcome * 1000

            # Additional benefit

            node_results[:, 24] = node_results[:, 23] - node_results[:, 22]

            # Save full results - DO NOT USUALLY USE!!!!!

            if data.save_node_results:
                filename = './' + data.output_location_node_results + \
                           str(i) + '.csv'
                node_df = pd.DataFrame()
                node_df['area'] = data.admissions_index.values
                node_df['admissions'] = data.admissions.values
                node_df['time_to_thrombolysis_unit'] = node_results[:, 7]
                node_df['IVT_unit_#'] = node_results[:, 8]
                node_df['time_to_thrombectomy_unit'] = node_results[:, 9]
                node_df['ET_unit_#'] = node_results[:, 10]
                node_df['Add_benefit_per_1000'] = node_results[:, 24]

                node_df = pd.merge(node_df,
                                   data.hospitals[['index_#', 'Hospital_name']],
                                   left_on='IVT_unit_#',
                                   right_on='index_#',
                                   how='left')

                node_df = node_df.rename(columns={'Hospital_name': 'IVT unit'})

                node_df = pd.merge(node_df,
                                   data.hospitals[['index_#', 'Hospital_name']],
                                   left_on='ET_unit_#',
                                   right_on='index_#',
                                   how='left')

                node_df = node_df.rename(columns={'Hospital_name': 'ET unit'})

                node_df.to_csv(filename, index=False)

            # Produce sceanrio results from node results

            # Result 0: Number of hospitals
            self.results[i, 0] = population[i,:].sum()

            # Result 1: mean time to thrombolysis

            self.results[i, 1] = (sum(node_results[:, 7] * data.admissions) /
                                  total_admissions)

            # Result 2: Maximum time to thrombolysis
            # Use mask where admissions >0
            mask = data.admissions > 0
            self.results[i, 2] = np.max(node_results[:, 7][mask])

            # Result 3: Mean time to thrombectomy
            self.results[i, 3] = (sum(node_results[:, 9] * data.admissions) /
                                  total_admissions)

            # Result 4: Maximum time to thrombectomy
            self.results[i, 4] = np.max(node_results[:, 9][mask])

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
            # one hospital. Create a new mask of units units if drip and
            # ship, otherwise use previous mask

            if not data.mothership:
                mask = data.thrombectomy_boolean

            admissions_to_used_units = thrombectomy_admissions_by_hospital[mask]

            self.results[i, 7] = np.min(admissions_to_used_units)
            self.results[i, 8] = np.max(admissions_to_used_units)

            # Result 9: Proportion patients within target thrombolysis time
            self.results[i, 9] = (sum(node_results[:, 15] * data.admissions)
                                  / total_admissions)

            # Result 10: Proportion patients attending unit with target first
            # admissions
            self.results[i, 10] = (sum(node_results[:, 16] * data.admissions)
                                   / total_admissions)

            # Result 11: Proportion patients meeting both thrombolysis targets
            self.results[i, 11] = (sum(node_results[:, 17] * data.admissions)
                                   / total_admissions)

            # Result 12: Proportion patients within target thrombectomy time
            self.results[i, 12] = (sum(node_results[:, 18] * data.admissions)
                                   / total_admissions)

            # Result 13: Proportion patients attending unit with target
            # thrombectomy admissions
            self.results[i, 13] = (sum(node_results[:, 19] * data.admissions)
                                   / total_admissions)

            # Result 14: Proportion patients meeting both thrombectomy targets
            self.results[i, 14] = (sum(node_results[:, 20] * data.admissions)
                                   / total_admissions)

            # Result 15: Proportion patients meeting all targets
            self.results[i, 15] = (sum(node_results[:, 21] * data.admissions)
                                   / total_admissions)

            # Result 16: 95th percentile time for thrombolysis
            self.results[i, 16] = self.calculate_weighted_percentiles(
                node_results[:, 7], data.admissions.values, [0.95])[0]

            # Result 17: 95th percentile time for thrombectomy
            self.results[i, 17] = self.calculate_weighted_percentiles(
                node_results[:, 9], data.admissions.values, [0.95])[0]

            # Result 18: Total transfers
            self.results[i, 18] = sum(node_results[:, 11])

            # Result 19: Total transfer time
            self.results[i, 19] = sum(node_results[:, 12])

            # Result 20: Net clinical outcome

            # Calculate clinical outcome for no treatment (per 1000 patients)
            self.results[i, 20] = (np.sum(node_results[:, 22] * admissions) /
                                   np.sum(admissions))

            # Outcomes with treatment (good outcomes per 1000 patients

            self.results[i, 21] = (np.sum(node_results[:, 23] * admissions) /
                                   np.sum(admissions))

            # Additional good outcomes per 1,000 admissions
            self.results[i, 22] = (np.sum(node_results[:, 24] * admissions) /
                                   np.sum(admissions))

            # Result 23 and 24 median times to thrombolysis and thrombectomy

            self.results[i, 23] = self.calculate_weighted_percentiles(
                node_results[:, 7], data.admissions.values, [0.5])[0]

            self.results[i, 24] = self.calculate_weighted_percentiles(
                node_results[:, 9], data.admissions.values, [0.50])[0]
            
            # Results 25-28 added clinical outcome ranges
            
            self.results[i, 25] = np.min(node_results[:, 24])

            self.results[i, 26] = self.calculate_weighted_percentiles(
                node_results[:, 24], data.admissions.values, [0.05])[0]
            
            self.results[i, 27] = self.calculate_weighted_percentiles(
                node_results[:, 24], data.admissions.values, [0.95])[0]
            
            self.results[i, 28] = np.max(node_results[:, 24])
            
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
