"""
Score population according to:
     0: Number of hospitals
     1: Average time to thrombolysis
     2: Maximum time to thrombolysis
     3: Average time to thrombectomy
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
    20: Net clinical outcome
"""

import numpy as np
import pandas as pd
import random as rn


class Pareto():
    def __init__(self, scores, criteria, min_pop, max_pop, pop):

        # Current scores
        self.scores = scores.results

        # Hold Pareto front
        self.pareto_front = np.array([])

        # Which results to use to selct Pareto front
        self.criteria = criteria

        # To hold processed scores for Pareto
        self.scores_for_pareto = []

        # Populations (unselected & all)
        unselected_population_ids = np.arange(self.scores.shape[0])
        all_population_ids = np.arange(self.scores.shape[0])
        population = pop.population

        # Change scores so all have higher is better
        self.preprocess_scores()

        crowding_count = 0

        while len(self.pareto_front) < min_pop:
            temp_pareto_front = self.identify_pareto(
                self.scores_for_pareto[unselected_population_ids, :],
                unselected_population_ids)

            # Check if latest pareto front needs reducing to not exceed maximum
            # Pareto front size
            if len(self.pareto_front) + len(temp_pareto_front) > max_pop:
                number_to_select = max_pop - len(self.pareto_front)

                temp_pareto_front = self.reduce_by_crowding(
                    temp_pareto_front, self.scores[temp_pareto_front],
                    number_to_select)

                crowding_count += 1
            # Add latest pareto front to full Pareto front
            self.pareto_front = np.hstack((self.pareto_front,
                                           temp_pareto_front))

            # Update unslected population ID by using sets to find IDs in all
            # ids that are not in the selected front

            unselected_set = set(all_population_ids) - set(self.pareto_front)

            unselected_population_ids = np.array(list(unselected_set))

        self.pareto_front_scores = self.scores[np.int_(self.pareto_front), :]

        # Update population and results to reflect chosen population
        pop.population = population[np.int_(self.pareto_front), :]

        scores.results = self.pareto_front_scores

        scores.hospital_first_admissions = scores.hospital_first_admissions[
                                           np.int_(self.pareto_front), :]

        scores.hospital_thrombectomy_admissions = \
            scores.hospital_thrombectomy_admissions[np.int_(
                self.pareto_front), :]

        return

    @staticmethod
    def calculate_crowding(scores):
        # Crowding is based on chrmosome scores (not chromosome binary values)
        # All scores are normalised between low and high. For any one score, all
        # solutions are sorted in order low to high. Crowding for chromsome x
        # for that score is the difference between the next highest and next
        # lowest score. Total crowding value sums all crowding for all scores

        population_size = len(scores[:, 0])
        number_of_scores = len(scores[0, :])

        # create crowding matrix of population (row) and score (column)
        crowding_matrix = np.zeros((population_size, number_of_scores))

        # normalise scores (ptp is max-min)
        normed_scores = (scores - scores.min(0)) / scores.ptp(0)

        # calculate crowding distance for each score in turn
        for col in range(number_of_scores):
            crowding = np.zeros(population_size)

            # end points have maximum crowding
            crowding[0] = 1
            crowding[population_size - 1] = 1

            # Sort each score (to calculate crowding between adjacent scores)
            sorted_scores = np.sort(normed_scores[:, col])

            sorted_scores_index = np.argsort(
                normed_scores[:, col])

            # Calculate crowding distance for each individual
            crowding[1:population_size - 1] = \
                (sorted_scores[2:population_size] -
                 sorted_scores[0:population_size - 2])

            # resort to orginal order (two steps)
            re_sort_order = np.argsort(sorted_scores_index)
            sorted_crowding = crowding[re_sort_order]

            # Record crowdinf distances
            crowding_matrix[:, col] = sorted_crowding

        # Sum croding distances of each score
        crowding_distances = np.sum(crowding_matrix, axis=1)

        return crowding_distances

    @staticmethod
    def identify_pareto(scores, population_ids):

        population_size = scores.shape[0]
        pareto_front = np.ones(population_size, dtype=bool)
        for i in range(population_size):
            for j in range(population_size):
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    # j dominates i
                    pareto_front[i] = 0
                    break

        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    def preprocess_scores(self):
        """Inverts scores where lower is better, so all scores have higher is
        better"""

        scores_for_pareto = np.copy(self.scores)
        # Pareto front assumes higher is better, therefore inverse scores where
        # lower is better. Add 1e-5 to avoid divide by zero error (e.g. no
        # patients in target)

        scores_for_pareto[:, 0] = 1 / (scores_for_pareto[:, 0] + 1e-5)
        scores_for_pareto[:, 1] = 1 / (scores_for_pareto[:, 1] + 1e-5)
        scores_for_pareto[:, 2] = 1 / (scores_for_pareto[:, 2] + 1e-5)
        scores_for_pareto[:, 3] = 1 / (scores_for_pareto[:, 3] + 1e-5)
        scores_for_pareto[:, 4] = 1 / (scores_for_pareto[:, 4] + 1e-5)
        scores_for_pareto[:, 6] = 1 / (scores_for_pareto[:, 6] + 1e-5)
        scores_for_pareto[:, 8] = 1 / (scores_for_pareto[:, 8] + 1e-5)
        scores_for_pareto[:, 16] = 1 / (scores_for_pareto[:, 16] + 1e-5)
        scores_for_pareto[:, 17] = 1 / (scores_for_pareto[:, 17] + 1e-5)
        scores_for_pareto[:, 18] = 1 / (scores_for_pareto[:, 18] + 1e-5)
        scores_for_pareto[:, 19] = 1 / (scores_for_pareto[:, 19] + 1e-5)

        self.scores_for_pareto = scores_for_pareto[:, self.criteria]

        return

    def reduce_by_crowding(self, population_ids, scores, number_to_select):
        # This function selects a number of solutions based on tournament of
        # crowding distances. Two members of the population are picked at
        # random. The one with the higher croding dostance is always picked

        crowding_distances = self.calculate_crowding(scores)

        picked_population_ids = np.zeros((number_to_select))

        picked_scores = np.zeros((number_to_select, len(scores[0, :])))

        for i in range(number_to_select):

            population_size = population_ids.shape[0]

            fighter1ID = rn.randint(0, population_size - 1)

            fighter2ID = rn.randint(0, population_size - 1)

            # If fighter # 1 is better
            if crowding_distances[fighter1ID] >= crowding_distances[
                fighter2ID]:

                # add solution to picked solutions array
                picked_population_ids[i] = population_ids[
                    fighter1ID]

                # Add score to picked scores array
                picked_scores[i, :] = scores[fighter1ID, :]

                # remove selected solution from available solutions
                population_ids = np.delete(population_ids, (fighter1ID),
                                           axis=0)

                scores = np.delete(scores, (fighter1ID), axis=0)

                crowding_distances = np.delete(crowding_distances, (fighter1ID),
                                               axis=0)
            else:
                picked_population_ids[i] = population_ids[fighter2ID]

                picked_scores[i, :] = scores[fighter2ID, :]

                population_ids = np.delete(population_ids, (fighter2ID), axis=0)

                scores = np.delete(scores, (fighter2ID), axis=0)

                crowding_distances = np.delete(
                    crowding_distances, (fighter2ID), axis=0)

        return (picked_population_ids)
