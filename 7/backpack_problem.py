import random


def generate_chromosome(r):
    return [random.randint(0, 1) for i in range(r)]


class backpack_problem:
    def __init__(self, backpack, weight, population_size=10):
        if len(backpack) != len(weight):
            raise ValueError("Niepoprawne dane!")
        self.backpack = backpack
        self.weight = weight
        self.id = [i - 1 for i in range(1, len(self.backpack) + 1)]


        self.population = [generate_chromosome(10) for i in range(population_size)]

    def fitness_function(self, chromosome):
        weight = 0
        counter = 0
        for bit in chromosome:
            if bit == 1:
                weight += self.weight[counter]
            counter += 1

        if weight > 35:
            return 0

        price = 0
        counter = 0
        for bit in chromosome:
            if bit == 1:
                price += self.backpack[counter]
            counter += 1

        return price

    def sort_population(self):
        self.population.sort(key=self.fitness_function, reverse=True)

    def sort_new_population(self, population, way):
        population.sort(key=self.fitness_function, reverse=way)

    def display_population(self):
        self.sort_population()
        for chromosome in self.population:
            print(chromosome, self.fitness_function(chromosome))

    def display_new_population(self, population):
        for chromosome in population:
            print(chromosome, self.fitness_function(chromosome))

    def cross(self, first, second):

        new_first = second[0:5] + first[5:10]
        new_second = first[0:5] + second[5:10]

        return [new_first, new_second]

    def roulette_wheel_selection(self):
        self.sort_population()
        temp_population = self.population.copy()
        new_population = []
        sections = []
        count = [10, 0]


        counter = 1
        for i in range(1, len(self.population)):
            sections.append(count[0])
            count[0] = self.fitness_function(self.population[counter]) + sections[-1]
            counter += 1

        while len(new_population) < 8:
            roullette = random.randint(0, sections[-1])
            for section in sections:
                if roullette < section:
                    index = (len(self.backpack) - 1) - sections.index(section)
                    new_population.append(temp_population[index - 2])
                    break

        for i in range(10):
            idx1, idx2 = random.randint(4, 7), random.randint(4, 7)
            new_chromosomes = self.cross(new_population[idx1], new_population[idx2])
            which_one = random.randint(0, 1)
            if which_one == 0:
                new_population[idx1] = new_chromosomes[0]
                new_population[idx2] = new_chromosomes[1]
            else:
                new_population[idx1] = new_chromosomes[1]
                new_population[idx2] = new_chromosomes[0]

        for i in range(len(new_population)):
            for j in range(len(new_population[0])):
                is_mutation = random.randint(1, 100)
                if is_mutation <= 5:
                    if new_population[i][j] == 1:
                        new_population[i][j] = 0
                    else:
                        new_population[i][j] = 1


        self.sort_population()
        self.sort_new_population(new_population, False)

        new_population[0] = self.population[0]
        new_population[1] = self.population[1]

        self.population = new_population
