import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
import sys

def create_grid(bounds: np.ndarray = np.array([[1, 10], [1, 10]]),
                 size: np.ndarray = np.array([10,15])) -> np.ndarray:
    """ создание прямоугольной неравномерной сетки """
    # a = np.zeros((np.shape(bounds)[0],1),dtype=np.ndarray)
    a = np.array(bounds[:,0],dtype=np.ndarray)
    for i in range(np.shape(bounds)[0]):
        a[i] = np.linspace(bounds[i, 0], bounds[i, 1], size[i])
    return a

def grid_of_funceval(value_grid: np.ndarray,
                      func) -> np.ndarray:
    """ только для пространсва D=2, для любой сетки в том числе и неравномерной"""
    idk = np.max(np.array([np.shape(value_grid[0])[0],np.shape(value_grid[1])[0]]))
    grid = np.full((idk,idk), np.inf)
    for i in range(np.shape(value_grid[0])[0]):
        for k in range(np.shape(value_grid[1])[0]):
            # grid[i,k] = func(value_grid[np.arange(0, np.shape(value_grid)[0]), qq])
            grid[i,k] = func(np.array([(value_grid[0])[i],(value_grid[1])[k]]))
    return grid

def min_of_function(grid: np.ndarray) -> np.ndarray:
    return [np.where(grid == np.min(grid))[0][0],\
            np.where(grid == np.min(grid))[1][0]],np.min(grid)

def find_in_grid(value_grid: np.ndarray,
                  solve: np.ndarray) -> np.ndarray:
    a = np.copy(solve)
    qq = np.random.randint(0, 1, np.shape(value_grid)[0])
    for i in range(np.shape(value_grid)[0]):
        hqd = np.absolute(solve[i] - (value_grid[i])[:])
        qq[i] = np.where(hqd == np.min(hqd))[0][0]
    return qq

def start_solve(value_grid: np.ndarray) -> np.ndarray:
    a = np.array([])
    index1 = np.array([],dtype=int)
    for i in range(np.shape(value_grid)[0]):
        index = np.random.randint(np.shape(value_grid[i])[0])
        a = np.append(a,(value_grid[i])[index])
        index1 = np.append(index1,index)
    return a, index1

def new_point_boltzman(bounds: np.ndarray,
                        value_grid: np.ndarray,
                        temperature: float,
                        index: np.ndarray) -> np.ndarray:
    std = np.sqrt(temperature) * np.ones(np.shape(bounds)[0])
    xc = np.random.normal(0, 1.0, size =  np.shape(bounds)[0])
    new_solve_index = index + xc * std
    new_solve = np.copy(new_solve_index)
    for i in range(np.shape(bounds)[0]):
        new_solve_index[i] = round(index[i] + xc[i] * std[i] * (np.shape((value_grid[i]))[0] - 1))
        while  new_solve_index[i] < 0 or  new_solve_index[i] > (np.shape((value_grid[i]))[0] - 1):
            xc[i] = np.random.normal(0, 1.0)
            new_solve_index[i] = round(index[i] + xc[i] * std[i] * (np.shape((value_grid[i]))[0] - 1))
    for i in range(np.shape(bounds)[0]):
        new_solve[i] = (value_grid[i])[new_solve_index.astype(int)[i]]
    return new_solve, new_solve_index.astype(int)

def find_start_temp(bounds: np.ndarray,
                    func) -> np.ndarray:
    low = bounds[:,0]
    high = bounds[:,1]
    fmax = np.finfo(float).min
    fmin = np.finfo(float).max
    for _ in range(50):
        start_solve = np.random.uniform(size = np.shape(bounds)[0]) * (high - low) + low
        start_solve_eval = func(start_solve)
        if start_solve_eval > fmax:
            fmax = start_solve_eval
        if start_solve_eval < fmin:
            fmin = start_solve_eval
            best_solve = start_solve
        temp = (fmax - fmin) * 1.5 
    return temp

def simulated_anelling(func,
                       bounds:np.ndarray,
                       value_grid: np.ndarray,
                       size_of_grid: np.ndarray,
                       iter_count:int = 100,
                       start_temp:float = 70,
                       flag:float = 0):
    if flag == 1:
        start_temp = find_start_temp(bounds,func)
        print('start temperature',start_temp)
    grid_of_values = np.full(shape = size_of_grid, fill_value = np.NAN)
    solve1, index1 = start_solve(value_grid)#global minimum
    energy1 = func(solve1)#value of global minimum
    grid_of_values[tuple(index1)] = energy1
    t = start_temp
    count1 = 0
    count = 1
    solve2, energy2, index2 = solve1, energy1, index1 #current_solve_of_alghorithm
    for i in range(iter_count):
        solve3, index3 = new_point_boltzman(bounds, value_grid, t,  index2)#new_solve_of_alghorithm
        if np.isnan(grid_of_values[tuple(index3)]):
            count +=1
            energy3 = func(solve3)
            grid_of_values[tuple(index3)] = energy3
        else:
            count1 +=1
            energy3 = grid_of_values[tuple(index3)]
        diff =  energy3 - energy2
        metropolis = np.exp(-diff / t)
        if energy3 < energy1:#difference w global_min and new_solve_of_alghorithm
            solve1, energy1, index1 = solve3, energy3, index3
            solve2, energy2, index2 = solve3, energy3, index3
        else:
            while(1):
                if energy3 < energy1:
                    solve1, energy1, index1 = solve3, energy3, index3
                    solve2, energy2, index2 = solve3, energy3, index3
                    break
                if np.random.uniform(0.0, 1.0) < metropolis:
                    solve2, energy2, index2 = solve3, energy3, index3
                    # print('loool')
                    break
                solve3, index3 = new_point_boltzman(bounds, value_grid, t,  index2)
                if np.isnan(grid_of_values[tuple(index3)]):
                    count +=1
                    energy3 = func(solve3)
                    grid_of_values[tuple(index3)] = energy3
                else:
                    count1 +=1
                    energy3 = grid_of_values[tuple(index3)]
                energy3 = func(solve3)
                diff =  energy3 - energy2
                metropolis = np.exp(-diff / t)  
        t = start_temp / np.log((i + 2) + 1)
    return [solve1, energy1, index1, count, count1]

def func2(x: np.ndarray):
    return -np.sin(0.5 * (x[0]**2) + - 0.25 * (x[1]**2) + 3) * np.cos(2*x[2] + 1 - np.exp(x[3]))

def func_3d(x: np.ndarray):
    return x[1]*np.sin(2*np.pi*x[0])+x[0]*np.cos(2*np.pi*x[1])

def try_function_anydemension_butwithout_exact_solution(
                it: int,
                temp: float,
                func,
                size_of_grid: float = np.array([1000,1000,1000,1000]),
                bounds: np.ndarray = np.array([[1, 10], [1, 10],[1, 10],[1, 10]])):
    zxc = create_grid(bounds, size_of_grid)
    ok, ok1, ok2, count, count1 = simulated_anelling(func, bounds, zxc, size_of_grid, it, temp)
    print(f'amount of points evalved:{count}')
    print(f'amount of saved computations:{count1}')
    print(f'numeric point:{ok}')
    print(f'numeric solution:{ok1}')
    print(f'numeric index in grid:{ok2}')

def try_function_anydemension_with_exact_solution(
                it: int,
                temp: float,
                func,
                size_of_grid: float = np.array([1000,1000]),
                bounds: np.ndarray = np.array([[1, 10], [1, 10]])):
    zxc = create_grid(bounds, size_of_grid)
    ret = dual_annealing(func,bounds=bounds)
    print(f'point with other numeric method(dual_anneling):{ret.x}')
    print(f'solution with other numeric method(dual_anneling):{ret.fun}')
    ok, ok1, ok2, count, count1 = simulated_anelling(func, bounds, zxc, size_of_grid, it, temp)
    print(f'amount of points evalved:{count}')
    print(f'amount of saved computations:{count1}')
    print(f'numeric point:{ok}')
    print(f'numeric solution:{ok1}')
    print(f'numeric index in grid:{ok2}')
    print(f'definition with exact solution:{(ok1 - ret.fun):.10f}')

def try_function_2d_demension(
                it: int,
                temp: float,
                func,
                size_of_grid: float = np.array([1000,1000]),
                bounds: np.ndarray = np.array([[1, 10], [1, 10]])):
    zxc = create_grid(bounds, size_of_grid)
    grid = grid_of_funceval(zxc, func)
    global point, point_eval
    point, point_eval = min_of_function(grid)
    print(f'exact index in grid:{point}')
    print(f'exact point:{[(zxc[0])[point[0]],(zxc[1])[point[1]]]}')
    print(f'exact solution:{point_eval}')
    ok, ok1, ok2, count, count1 = simulated_anelling(func, bounds, zxc, size_of_grid, it, temp)
    print(f'amount of points evalved:{count}')
    print(f'amount of saved computations:{count1}')
    print(f'numeric point:{ok}')
    print(f'numeric solution:{ok1}')
    print(f'numeric index in grid:{ok2}')
    print(f'definition with exact solution:{(ok1 - point_eval):.10f}')

def try_function_2d_demension_withnumericmethod(
                it: int,
                temp: float,
                func,
                size_of_grid: float = np.array([1000,1000]),
                bounds: np.ndarray = np.array([[1, 10], [1, 10]])):
    zxc = create_grid(bounds, size_of_grid)
    ret = dual_annealing(func,bounds=bounds)
    print(f'point with other numeric method(dual_anneling):{ret.x}')
    print(f'solution with other numeric method(dual_anneling):{ret.fun}')
    ok, ok1, ok2, count, count1 = simulated_anelling(func, bounds, zxc, size_of_grid, it, temp)
    print(f'amount of points evalved:{count}')
    print(f'amount of saved computations:{count1}')
    print(f'numeric point:{ok}')
    print(f'numeric solution:{ok1}')
    print(f'numeric index in grid:{ok2}')
    print(f'definition with exact solution:{(ok1 - ret.fun):.10f}')

def try_function_2d_demension_100tests(it: int,
                                       temp: float,
                                       func,
                                       size_of_grid: np.ndarray = np.array([200,100]), 
                                       bounds: np.ndarray = np.array([[1, 10], [1, 10]]), 
                                       flag:float = 0):
    zxc = create_grid(bounds, size_of_grid)
    grid = grid_of_funceval(zxc, func)
    global point, point_eval
    point, point_eval = min_of_function(grid)
    print('exact index in grid', point)
    print('exact point',[(zxc[0])[point[0]],(zxc[1])[point[1]]])
    print('exact solution:', point_eval)
    # print(np.min(zxc1))
    count = 0
    for i in range(100):
        ok,ok1,ok2,ok3,ok4 = simulated_anelling(func, bounds, zxc, size_of_grid, it, temp)
        print(f"numeric index:{find_in_grid(zxc,ok)[0]}~{find_in_grid(zxc,ok)[1]} exact:{point[0]}~{point[1]}")
        if find_in_grid(zxc,ok)[0] - point[0] == 0 and find_in_grid(zxc,ok)[1] - point[1] == 0:
            count += 1 
    print('count of exact solutions',count)

