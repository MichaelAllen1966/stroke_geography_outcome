import numpy as np
import random as rn


class Pop():
    """
    Pop is the population class. It holds the current population and methods
    used for generating populations, including breeding.
    """

    def __init__(self):
        self.population = []  # holds the current population
        return

    def create_random_population(self, rows_required, hospitals, fix_hospitals):
        """
        Create a random population.
        Note: population size returned may be smaller than requested due to
        removal of empty or non-unique rows. Method to 'top up' population
        removed as this is not necessary.
        """

        hospital_count = len(hospitals)
        population = np.zeros((rows_required, hospital_count))

        # Add in random number of hospitals and shuffle
        for i in range(rows_required):
            x = rn.randint(1, hospital_count)
            population[i, 0:x] = 1
            np.random.shuffle(population[i])

        # Fix open and closed hospitals if required
        if fix_hospitals:
            population = self.fix_hospital_status(hospitals, population)

        # Remove any rows with no hospitals
        check_hospitals = np.sum(population, axis=1) > 0
        population = population[check_hospitals, :]

        # Remove non-unique rows
        population = np.unique(population, axis=0)

        return population

    @staticmethod
    def crossover(parents, max_crossover_points):

        chromsome_length = parents.shape[1]

        number_crossover_points = rn.randint(1, max_crossover_points)

        # pick random crossover points in gene, avoid zero position
        crossover_points = rn.sample(range(1, chromsome_length), \
                                     number_crossover_points)

        # appended zero at front for calulation of interval to first crossover
        crossover_points = np.append([0], np.sort(crossover_points))

        # create intervals of ones and zeros
        intervals = crossover_points[1:] - crossover_points[:-1]

        # Add last interval
        intervals = np.append(
            [intervals], [chromsome_length - np.amax(crossover_points)])

        # Build boolean arrays for cross-overs.  Sub sections will be made up
        # of repeats ofboolean true or false, start with true
        current_bool = True

        # empty list required for append
        selection1 = []

        # interval is the interval between crossoevrs (stored in 'intervals')
        for interval in intervals:
            # create subsection of true or false
            new_section = np.repeat(current_bool, interval)

            # swap true to false and vice versa
            current_bool = not current_bool

            # add the new section to the existing array
            selection1 = np.append([selection1], [new_section])

        selection1 = np.array([selection1], dtype=bool)

        # invert boolean selection for second cross-over product
        selection2 = np.invert(selection1)

        # choose from parents based on selection vector
        child_1 = np.choose(selection1, parents)
        child_2 = np.choose(selection2, parents)

        children = np.append(child_1, child_2, axis=0)

        return children

    @staticmethod
    def fix_hospital_status(hospitals, population):
        """
        Fixes hospitals to be forced open or forced closed as required.
        This is done by overlaying a matrix of forced open (1) or forced closed
        (-1)
        """
        fix_list = hospitals['Fixed'].values

        # Set fixed CSC (labelled as 2 in list) to a value of 1 
        fix_list[fix_list == 2] = 1

        population_size = population.shape[0]
        fix_matrix = np.array([fix_list, ] * population_size)

        # Fix the forced open HASUs to have a value 1
        population[fix_matrix == 1] = 1

        # Fix the closed hospitals to have a value 0
        population[fix_matrix == -1] = 0
        return population

    def generate_child_population(
            self, maximum_crossovers, mutation, hospitals, fix_hospitals):


        pop_required = self.population.shape[0]
        hospital_count = self.population.shape[1]
        child_population = np.zeros((0, self.population.shape[1]))
        population_size = self.population.shape[0]

        for i in range(int((pop_required/2)+1)):
            # Select parents at random
            parent1_ID = rn.randint(0, population_size - 1)
            parent2_ID = rn.randint(0, population_size - 1)

            # Stack parents one above the other
            parents = np.vstack((self.population[parent1_ID],
                                 self.population[parent2_ID]))

            children = self.crossover(parents, maximum_crossovers)

            child_population = np.vstack((child_population, children))

        # Apply random mutation
        random_mutation_array = np.random.random(
            size=(child_population.shape))
        
        random_mutation_boolean = \
            random_mutation_array <= mutation

        child_population[random_mutation_boolean] = \
            np.logical_not(child_population[random_mutation_boolean])
	
	# Fix hospital status if required
        if fix_hospitals:
            child_population = \
                self.fix_hospital_status(hospitals, child_population)

        # Remove any rows with no hospitals
        check_hospitals = np.sum(child_population, axis=1) > 0
        child_population = child_population[check_hospitals, :]

        return child_population
