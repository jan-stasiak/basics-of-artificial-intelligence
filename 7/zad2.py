import math
import random


class Zad2:
    def __init__(self, how_many=10, size=10, user_range=(0, 1)):
        self.population = [self.generate_chromosome(size, user_range) for i in range(how_many)]
        self.how_many = how_many
        self.size = size

    def generate_chromosome(self, how_many, user_range):
        return [random.randint(user_range[0], user_range[1]) for i in range(how_many)]

    def display_population(self):
        self.sort_population()
        for chromosome in self.population:
            print(chromosome, self.bin_to_dec(chromosome), self.fitness_function(chromosome))

    def bin_to_dec(self, chromosome):
        sum_a = 0
        count = 0
        for bit in range(int(self.size / 2) - 1, -1, -1):
            sum_a += chromosome[bit] * pow(2, count)
            count += 1

        sum_b = 0
        count = 0
        for bit in range(self.size - 1, int(self.size / 2) - 1, -1):
            sum_b += chromosome[bit] * pow(2, count)
            count += 1

        return sum_a, sum_b

    def fitness_function(self, chromosome):
        a, b = self.bin_to_dec(chromosome)
        return abs((2 * pow(a, 2)) + b - 33)

    def sort_population(self):
        self.population.sort(key=self.fitness_function)

    # 10, 20, 40, 80, 160, 320, 640, 1280
    def roulette_wheel_selection(self):
        self.sort_population()
        new_population = []
        sections = []
        count = [self.fitness_function(self.population[0])]
        counter = 1
        for i in range(1, len(self.population)):
            sections.append(count[0])
            count[0] = (1 / (self.fitness_function(self.population[counter]) + 1))
            counter += 1


        new_population.append(self.population[0])
        new_population.append(self.population[1])


        self.population.sort(key=self.fitness_function, reverse=True)
        temp = sum(sections)
        while len(new_population) < 10:
            roullette = random.uniform(0, temp)
            for section in sections:
                if roullette < section:
                    index = (self.how_many - 3) - sections.index(section)
                    new_population.append(self.population[index - 1])
                    break

        self.population = new_population

    def cross(self):
        self.sort_population()
        for i in range(self.how_many - 1, int(self.how_many / 2) - 1, -1):
            new_first = self.population[i - 1][0:4] + self.population[i][4:8]
            new_second = self.population[i][0:4] + self.population[i - 1][4:8]
            which_one = random.randint(0, 1)
            if which_one == 0:
                which_one = random.randint(0, 1)
                if which_one == 0:
                    self.population[i] = new_first
                else:
                    self.population[i] = new_second
            else:
                which_one = random.randint(0, 1)
                if which_one == 0:
                    self.population[i - 1] = new_first
                else:
                    self.population[i - 1] = new_second



        return new_first, new_second


    def mutation(self):
        for chromosome in self.population:
            is_mutation = random.randint(1, 10)
            if is_mutation == 1:
                gene = random.randint(0, self.size - 1)
                if chromosome[gene] == 1:
                    chromosome[gene] = 0
                else:
                    chromosome[gene] = 1


    def find_solution(self):
        iteration = 0
        while(True):
            self.sort_population()
            # self.display_population()
            if self.fitness_function(self.population[0]) == 0:
                break
            self.roulette_wheel_selection()
            self.cross()
            self.mutation()
            iteration += 1
        self.display_population()
        return iteration
