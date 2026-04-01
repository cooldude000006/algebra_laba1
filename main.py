import numpy as np
import time
import sys

'''ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ'''
def set_seed(seed=42):
    np.random.seed(seed)

def generate_random_matrix(n, low = -1.0, high = 1.0):
        A = np.random.uniform(low, high, (n, n))
        b = np.random.uniform(low, high, n)
        return A, b

def generate_hilbert_matrix(n): #H[i,j] = 1 / (i + j - 1)
      H=np.zeros((n,n))
      for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i+j+1)
      return H

def calculate_residual(A, x, b): #||Ax - b||
     return np.linalg.norm(A @ x - b)

def calculate_relative_error(x_true, x_approx): #||x_approx - x_true|| / ||x_true||
     norm_true = np.linalg.norm(x_true)
     if norm_true == 0:
          return np.linalg.norm(x_approx)
     return np.linalg.norm(x_approx - x_true) / norm_true

def measure_time(func, *args):
     start = time.perf_counter()
     result = func(*args)
     end = time.perf_counter()
     return end - start, result

'''РЕАЛИЗАЦИЯ АЛГОРИТМОВ РЕШЕНИЯ СЛАУ'''
def gaussian_elimination_no_pivot(A, b): #м. Гаусса без выбора ведущего элемента
     n = len(b)
     A = A.astype(float).copy()
     b = b.astype(float).copy()

     for k in range(n-1):
          if abs(A[k, k]) < 1e-12:
               continue
          for i in range(k+1, n):
               factor = A[i, k] / A[k, k]
               A[i, k:] -= factor*A[k, k:]
               b[i] -= factor*b[k]
     x = np.zeros(n)
     for i in range(n-1, -1, -1):
          if abs(A[i, i]) < 1e-12:
               x[i] = 0.0
          else:
               x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i,i]
     return x

def gaussian_elimination_pivot(A, b): #м. Гаусса с выбором главного элемента
     n = len(b)
     A = A.astype(float).copy()
     b = b.astype(float).copy()          

     for k in range(n-1):
          max_idx = k + np.argmax(np.abs(A[k:, k]))
          if abs(A[max_idx, k]) < 1e-12:
               continue
          if max_idx != k:
               A[[k, max_idx]] = A[[max_idx, k]]
               b[[k, max_idx]] = b[[max_idx, k]]
          for i in range(k+1, n):
               factor = A[i, k] / A[k, k]
               A[i, k:] -= factor * A[k, k:]
               b[i] -= factor * b[k]
     x = np.zeros(n)
     for i in range(n-1, -1, -1):
          if abs(A[i, i]) < 1e-12:
               x[i] = 0.0
          else:
               x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
     return x

def lu_decomposition(A): #LU-разложение без перестановок (A = L * U)
     n = A.shape[0]
     L = np.eye(n)
     U = A.astype(float).copy()

     for k in range(n-1):
          if abs(U[k, k]) < 1e-12:
               continue
          for i in range(k+1, n):
               factor = U[i, k] / U[k, k]
               L[i, k] = factor
               U[i, k:] -= factor * U[k, k:]
               U[i, k] = 0.0
     return L, U 

def forward_substitution(L, b): #Прямая подстановка для решения L*y = b
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        if abs(L[i, i]) < 1e-12: 
            y[i] = 0.0
        else:
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def backward_substitution(U, y): #Обратная подстановка для решения U*x = y
     n = len(y)
     x = np.zeros(n)
     for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            x[i] = 0.0
        else:
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
     return x

def solve_lu(L, U, b): #Решение СЛАУ через LU-разложение
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

'''ЭКСПЕРИМЕНТЫ'''
def experiment_4_1(): #Сравнение времени решения одной системы
    print("\n Эксперимент 4.1: Сравнение времени решения одной системы ")
    sizes = [100, 200, 500, 1000]
    print(f"{'N':<6} | {'Gauss No Pivot':<15} | {'Gauss Pivot':<15} | {'LU Total':<15} | {'LU Decomp':<15} | {'LU Solve':<15}")
    print("-" * 95)
    
    results = []
    for n in sizes:
        set_seed(42)
        A, b = generate_random_matrix(n)
        
        # Gauss No Pivot
        t_gp, _ = measure_time(gaussian_elimination_no_pivot, A, b)
        
        # Gauss Pivot
        t_gpivot, _ = measure_time(gaussian_elimination_pivot, A, b)
        
        # LU
        t_decomp, (L, U) = measure_time(lu_decomposition, A)
        t_solve, _ = measure_time(solve_lu, L, U, b)
        t_lu_total = t_decomp + t_solve
        
        print(f"{n:<6} | {t_gp:<15.6f} | {t_gpivot:<15.6f} | {t_lu_total:<15.6f} | {t_decomp:<15.6f} | {t_solve:<15.6f}")
        results.append((n, t_gp, t_gpivot, t_lu_total))
    return results

def experiment_4_2(): #Экономия времени при множественных правых частях
    print("\n Эксперимент 4.2: Множественные правые части (N=500) ")
    n = 500
    ks = [1, 10, 100]
    set_seed(42)
    A, _ = generate_random_matrix(n)
    
    # Предварительное LU разложение (один раз)
    t_decomp, (L, U) = measure_time(lu_decomposition, A)
    
    print(f"{'K':<6} | {'Gauss Total':<15} | {'LU Total (Decomp+K*Solve)':<25}")
    print("-" * 60)
    
    for k in ks:
        set_seed(42 + k) # Разные seed для разных наборов b
        b_list = [np.random.uniform(-1, 1, n) for _ in range(k)]
        
        # Gauss Pivot для всех k
        start = time.perf_counter()
        for b in b_list:
            gaussian_elimination_pivot(A, b)
        t_gauss_total = time.perf_counter() - start
        
        # LU для всех k (разложение уже есть, считаем только решения + время разложения один раз)
        start = time.perf_counter()
        for b in b_list:
            solve_lu(L, U, b)
        t_lu_solves = time.perf_counter() - start
        t_lu_total = t_decomp + t_lu_solves
        
        print(f"{k:<6} | {t_gauss_total:<15.6f} | {t_lu_total:<25.6f}")

def experiment_4_3(): #Проверка точности на плохо обусловленных матрицах (Гильберт)
    print("\n Эксперимент 4.3: Точность на матрице Гильберта ")
    sizes = [5, 10, 15]
    print(f"{'N':<6} | {'Method':<20} | {'Rel. Error':<15} | {'Residual':<15}")
    print("-" * 65)
    
    for n in sizes:
        H = generate_hilbert_matrix(n)
        x_true = np.ones(n)
        b = H @ x_true
        
        methods = [
            ("Gauss No Pivot", gaussian_elimination_no_pivot),
            ("Gauss Pivot", gaussian_elimination_pivot),
            ("LU Decomposition", lambda A, b: solve_lu(*lu_decomposition(A), b))
        ]
        
        for name, func in methods:
            try:
                x_approx = func(H, b)
                rel_err = calculate_relative_error(x_true, x_approx)
                resid = calculate_residual(H, x_approx, b)
                print(f"{n:<6} | {name:<20} | {rel_err:<15.6e} | {resid:<15.6e}")
            except Exception as e:
                print(f"{n:<6} | {name:<20} | {'Error':<15} | {str(e):<15}")

'''MAIN'''
if __name__ == "__main__":
    print("Запуск лабораторной работы: Сравнение методов решения СЛАУ")
    experiment_4_1()
    experiment_4_2()
    experiment_4_3()
    print("\nРабота завершена. Результаты выше есть в отчёте")
