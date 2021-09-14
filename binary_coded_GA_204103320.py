import random
import math
import itertools
import matplotlib.pyplot as plt
           


def function(x1,x2):
    f = 1/(x1+x2-(2*pow(x1,2))-pow(x2,2)+(x1*x2)+1)
    return(f)


def zero_one_rand():
    a = random.randint(0,1)
    return(a)

def gen_one_str(l):
    gene = []
    for i in range(1,l+1):
        bit = zero_one_rand()
        gene.append(bit)                  #final gene = chromosome
    return(gene)

def decoded_x(test_list):
    res = int("".join(str(x) for x in test_list), 2)
    return(res)

def real_x(x_min,x1_max,l,xd):
    xr = x_min + (((x1_max-x1_min)/(pow(2,l)-1))*(xd))
    return(xr)

                                            #roulette wheel
def roulette_wheel_pop(population_rou, probabilities, number):
    chosen = []
    for n in range(number):
        r = random.random()
        for (i, individual) in enumerate(population_rou):
            if r <= probabilities[i]:
                chosen.append(list(individual))
                break
    return chosen
                                #TWO POINT CROSSOVER FUNCTION 
def Crossover(parent_1,parent_2,points):
    p1 = parent_1
    p2 = parent_2 
    for i in range(points[0],points[1]):
        p1[i],p2[i] = p2[i],p1[i]       #swap the genetic information
    p1,p2 = list(p1),list(p2) 
    for i in range(points[1],len(p1)):
        p1[i],p2[i] = p1[i],p2[i]    
    p1,p2 = list(p1),list(p2) 
    return p1,p2

                                # MUTATION
def Mutation(child,m_c):
    for h in range(0,(len(child))):
        r_mc = random.random()
        if r_mc<=m_c:
            child[h] = abs(child[h]-1) 
    return child
                                        # length finder depending on accuracy
def lengthfinder(Xmin,Xmax,ep):
        d = (Xmax-Xmin)/ep
        leng = math.log2(d)
        return math.ceil(leng)

                                #GA DESCRIPTION
N = int(input("Enter The Population Size: "))
pc = float(input("Enter The Crossover probability: "))
mc = float(input("Enter The Mutation probability: "))
# epsi = float(input("Enter The accuracy you required : "))
x1_min = 0
x2_min = 0
x1_max = 0.5
x2_max = 0.5
# l1 = lengthfinder(x1_min,x1_max,epsi)
# print(l1)
l1 = int(input("Enter The no of bits (length) of varialble x1 : "))
len_x1 = l1
# l2 = lengthfinder(x2_min,x2_max,epsi)
# print(l2)
l2 = int(input("Enter The no of bits (length) of varialble x2 : "))
len_x2 = l2
tot_len = len_x1+len_x2
GEN = 0
MAX_GEN = 70

                           #generation of population gen = 0
average_fitness_list = []
generation_list = []
max_of_fitness = []
min_of_fitness = []
optimal_solution_x1 = []
optimal_solution_x2 = []
gen_wise_x1 = {}
gen_wise_x2 = {}
population = {}
for j in range(1,N+1):
    population[j] = gen_one_str(tot_len)

for g in range(GEN,MAX_GEN):

    population_list = list(population.values())
                                    #function evaluations
    generation_list.append(g)
    fitness_eval = []
    x1_list = []
    x2_list = []
    di = {}
    for key,value in population.items():
        x1d = decoded_x(value[0:len_x1])
        x1r = real_x(x1_min,x1_max,len_x1,x1d)
        x1_list.append(x1r)
        x2d = decoded_x(value[len_x1:])
        x2r = real_x(x2_min,x2_max,len_x2,x2d)
        x2_list.append(x2r)
        fun = function(x1r,x2r)
        fitness_eval.append(fun)
        
    average_fitness = sum(fitness_eval)/len(fitness_eval)
    average_fitness_list.append(average_fitness)
    max_of_fitness.append(max(fitness_eval))
    min_of_fitness.append(min(fitness_eval))
    m = len(fitness_eval)   

    osx1 = x1_list[fitness_eval.index(max(fitness_eval))]
    osx2 = x2_list[fitness_eval.index(max(fitness_eval))]   

    optimal_solution_x1.append(osx1)
    optimal_solution_x2.append(osx2)

    gen_wise_x1[g] = x1_list
    gen_wise_x2[g] = x2_list
                                            #REPRODUCTION
    k = (1/sum(fitness_eval))
    prob_fitness = [i * k  for i in fitness_eval]
    probab = [sum(prob_fitness[:i+1]) for i in range(len(prob_fitness))]

    #to include ranking selection only prob_fitness needs to be changed(ranking is applied on prob_fitness)
                                            #mating pool
    roul = roulette_wheel_pop(population_list, probab, N)
    matingpool_chromosome = {}
    for j in range(1,N+1):
        matingpool_chromosome[j] = roul[j-1]

                                                #CROSSOVER
    matingpool_copy = matingpool_chromosome.copy()
    new_population = {}
    for i in range(1,int((N/2)+1)):
        r1 = random.choice(list(matingpool_copy.keys()))
        matingpool_copy.pop(r1)
        r2 = random.choice(list(matingpool_copy.keys()))
        matingpool_copy.pop(r2)
        r_pc = random.random()
        if r_pc<=pc:
            r_2point = random.sample(range(1,tot_len),2)
            r_2point.sort()
            parent1 = matingpool_chromosome.get(r1)
            parent2 = matingpool_chromosome.get(r2)
            children1,children2 = Crossover(parent1,parent2,r_2point)
            offspring1 = Mutation(children1,mc)
            offspring2 = Mutation(children2,mc)
            new_population[r1] = offspring1
            new_population[r2] = offspring2

        else:
            parent1 = matingpool_chromosome.get(r1)
            parent2 = matingpool_chromosome.get(r2)
            new_population[r1] = parent1
            new_population[r2] = parent2

    new_population_list = list(new_population.values())
    population2N = population_list + new_population_list
    for j in range(1,N+1):
        population[j] = new_population_list[j-1]
    
                                                    # survival of the fittest
    
    for key,value in population.items():
        x1d = decoded_x(value[0:len_x1])
        x1r = real_x(x1_min,x1_max,len_x1,x1d)
        x2d = decoded_x(value[len_x1:])
        x2r = real_x(x2_min,x2_max,len_x2,x2d)
        fun = function(x1r,x2r)
        fitness_eval.append(fun)

    pop_sol_fitness_eval_dict = dict(zip(fitness_eval, population2N))
    new_pop_sol_fitness_eval_dict = {}
    sortedList=sorted(pop_sol_fitness_eval_dict.keys(),reverse=True)

    for sortedvalue in sortedList:
        for ke, valu in pop_sol_fitness_eval_dict.items():
            if ke==sortedvalue:
                new_pop_sol_fitness_eval_dict[ke]=valu

    population.clear()
    population = dict(itertools.islice(new_pop_sol_fitness_eval_dict.items(), N))
                                         # sending new solutions as parent solution
    new_population_list = list(population.values())
    population.clear()
    for j in range(1,N+1):
        population[j] = new_population_list[j-1]

                                    # loop for generation updation is over

                                    # average fitness graph
plt.rc('font',family='Times New Roman',weight = 1.3)
plt.plot(generation_list,average_fitness_list,label = "Average fitness")
plt.legend(loc = 'center right')
plt.xlabel('Number of Generations', fontweight='bold')
plt.ylabel('Fitness value', fontweight='bold')
plt.title("Average Fitness value v/s Generations", fontweight='bold')
plt.savefig('Average fitness graph.jpg',bbox_inches = 'tight',dpi=200)
plt.show()

                                        # max and min fitness graph

plt.plot(generation_list,min_of_fitness,label = "Minimum")
plt.plot(generation_list,max_of_fitness,label = "Maximum")
# plt.plot(generation_list,average_fitness_list,label = "Average")
plt.legend(title = "Fitness", loc='lower right')
plt.tight_layout()
plt.xlabel('Number of Generations', fontweight='bold')
plt.ylabel('Fitness value', fontweight='bold')
plt.title("Fitness values v/s Generations", fontweight='bold')
plt.savefig('Maximum and minimum fitness graph.jpg',bbox_inches = 'tight',dpi=200)
plt.show()


                            # optimal solution graph for x1

plt.plot(generation_list,gen_wise_x1.values(), marker='o',markersize = 3,linewidth=0)
plt.xlabel('Number of Generations', fontweight='bold')
plt.ylabel('x1', fontweight='bold')
plt.title("x1 values of total Population v/s Generation", fontweight='bold')
plt.savefig('optimal solution graph for x1.jpg',bbox_inches = 'tight',dpi=200)
plt.show()

                            # optimal solution graph for x2
plt.plot(generation_list,gen_wise_x2.values(), marker='o',markersize = 3,linewidth=0)
plt.xlabel('Number of Generations', fontweight='bold')
plt.ylabel('x2', fontweight='bold')
plt.title("x2 values of total Population v/s Generation", fontweight='bold')
plt.savefig('optimal solution graph for x2.jpg',bbox_inches = 'tight',dpi=200)
plt.show()
print("Binary Coded Genetic Algorithm has been performed and")
print("the plotted graphs have been saved in the current folder")
