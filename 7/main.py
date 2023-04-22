import random
import zad2 as function
import backpack_problem as backpack_class


def fitness_function(chromosome):
    return chromosome.count(1)


def generate_chromosome(r):
    return [random.randint(0, 1) for i in range(r)]


def cross(first, second):
    place = random.randint(0, 10)

    new_first = first[0:place] + second[place:10]
    new_second = second[0:place] + first[place:10]
    return new_first, new_second


def zad1():
    population = [generate_chromosome(10) for i in range(10)]

    i = 0
    while (True):
        print(f"Generation: {i}")
        population.sort(key=fitness_function)
        if fitness_function(population[9]) == 10:
            break
        for chromosome in population:
            print(chromosome, fitness_function(chromosome))
        print()
        new_first, new_second = cross(population[8], population[9])
        population[0] = new_first
        population[1] = new_second

        posibility = random.randint(0, 10)
        if posibility <= 6:
            gene = random.randint(0, 9)
            which_chromosome = random.randint(8, 9)
            if population[which_chromosome][gene] == 0:
                population[which_chromosome][gene] = 1
            else:
                population[which_chromosome][gene] = 0
        i += 1
    print(f"Generation {i}:")
    for chromosome in population:
        print(chromosome, fitness_function(chromosome))


def zad2():
    how_many = 0
    sum = 0
    max = float("-inf")
    population = function.Zad2(10, 8, (0, 1))
    temp = population.find_solution()
    print(f"Generation {temp}")


def zad3():


    for i in range(1):
        backpack = [266, 442, 671, 526, 388, 245, 210, 145, 126, 322]
        weight = [3, 13, 10, 9, 7, 1, 8, 8, 2, 9]
        problem = backpack_class.backpack_problem(backpack, weight, 8)

        iteration = 0
        while True:
            problem.roulette_wheel_selection()
            problem.sort_population()
            iteration += 1
            if problem.fitness_function(problem.population[0]) == 2222:
                problem.display_population()
                break

        print(f"Problem was solved within {iteration} iterations")


def main():
    # zad1()
    # zad2()
    zad3()


if __name__ == "__main__":
    main()
