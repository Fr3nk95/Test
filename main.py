from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import time
from dataclasses import dataclass
import random
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Integer, cqm_to_bqm, SimulatedAnnealingSampler
from collections import deque
import heapq
from itertools import combinations
import uvicorn

app = FastAPI(title="Portfolio Optimization API")

# Data models for API
class InvestmentOption(BaseModel):
    name: str
    a: float
    b: float
    c: float

class Antisynergy(BaseModel):
    option1: str
    option2: str
    value: float

class PortfolioRequest(BaseModel):
    capitale: int
    unita_capitale: int
    opzioni: List[InvestmentOption]
    antisinergie: List[Antisynergy]
    penalty_strength: float = 1.0

class SolutionResult(BaseModel):
    method_name: str
    energy: float
    sample: Dict[Any, int]
    computation_time: float
    iterations: int
    num_variables: int

class PortfolioResponse(BaseModel):
    results: List[SolutionResult]
    best_solution: Dict[str, Any]
    qubo_matrix: List[List[float]]
    qubo_variables: List[str]
    qubo_offset: float

# Copy all your solver classes here (BQMSolver, etc.)
@dataclass
class BQMSolutionResult:
    method_name: str
    energy: float
    sample: Dict[Any, int]
    computation_time: float
    iterations: int
    num_variables: int

class BQMSolver:
    """Base class for BQM solvers"""
    def __init__(self, bqm: BinaryQuadraticModel, seed: Optional[int] = None):
        self.bqm = bqm
        self.variables = list(bqm.variables)
        self.num_vars = len(self.variables)
        self.var_to_idx = {var: idx for idx, var in enumerate(self.variables)}
        self.idx_to_var = {idx: var for idx, var in enumerate(self.variables)}
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _sample_to_array(self, sample: Dict) -> np.ndarray:
        """Convert sample dict to numpy array"""
        array = np.zeros(self.num_vars, dtype=int)
        for var, value in sample.items():
            if var in self.var_to_idx:
                array[self.var_to_idx[var]] = value
        return array
    
    def _array_to_sample(self, array: np.ndarray) -> Dict:
        """Convert numpy array to sample dict"""
        return {self.idx_to_var[i]: int(array[i]) for i in range(len(array))}
    
    def _random_sample(self) -> Dict:
        """Generate random binary sample"""
        return {var: np.random.randint(0, 2) for var in self.variables}
    
    def _compute_flip_delta(self, sample_array: np.ndarray, var_idx: int) -> float:
        """Compute energy change when flipping variable at var_idx"""
        var = self.idx_to_var[var_idx]
        current_value = sample_array[var_idx]
        
        # Linear contribution
        delta = (1 - 2 * current_value) * self.bqm.linear.get(var, 0.0)
        
        # Quadratic contributions
        for neighbor_var, weight in self.bqm.adj[var].items():
            if neighbor_var in self.var_to_idx:
                neighbor_idx = self.var_to_idx[neighbor_var]
                neighbor_value = sample_array[neighbor_idx]
                delta += (1 - 2 * current_value) * weight * neighbor_value
        
        return delta

# Add a few key solvers (you can add more as needed)
class BQMSimulatedAnnealingSolver(BQMSolver):
    def __init__(self, bqm: BinaryQuadraticModel, initial_temp: float = 10.0, 
                 final_temp: float = 1e-3, max_iterations: int = 2000, 
                 schedule: str = "exponential", seed: Optional[int] = None):
        super().__init__(bqm, seed)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.schedule = schedule
        self.name = "BQM_SimulatedAnnealing"
    
    def _temperature(self, iteration_frac: float) -> float:
        """Compute temperature at given iteration fraction"""
        if self.schedule == "exponential":
            return self.initial_temp * (self.final_temp / self.initial_temp) ** iteration_frac
        elif self.schedule == "linear":
            return self.initial_temp + (self.final_temp - self.initial_temp) * iteration_frac
        else:
            return self.initial_temp * (self.final_temp / self.initial_temp) ** iteration_frac
    
    def solve(self) -> BQMSolutionResult:
        start_time = time.time()
        
        # Initialize
        current_sample = self._random_sample()
        current_array = self._sample_to_array(current_sample)
        current_energy = self.bqm.energy(current_sample)
        
        best_sample = current_sample.copy()
        best_energy = current_energy
        
        for iteration in range(1, self.max_iterations + 1):
            # Temperature schedule
            t_frac = iteration / float(self.max_iterations)
            temperature = max(self._temperature(t_frac), 1e-10)
            
            # Random variable to flip
            var_idx = np.random.randint(0, self.num_vars)
            delta = self._compute_flip_delta(current_array, var_idx)
            
            # Metropolis criterion
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                # Accept move
                current_array[var_idx] = 1 - current_array[var_idx]
                current_sample = self._array_to_sample(current_array)
                current_energy += delta
                
                # Update best
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sample = current_sample.copy()
        
        elapsed_time = time.time() - start_time
        return BQMSolutionResult(
            method_name=self.name,
            energy=best_energy,
            sample=best_sample,
            computation_time=elapsed_time,
            iterations=self.max_iterations,
            num_variables=self.num_vars
        )

class DimodAnnealingSolver(BQMSolver):
    """Simulated Annealing using dimod's SimulatedAnnealingSampler"""

    def __init__(self, bqm: BinaryQuadraticModel, num_reads: int = 100, seed: int = None):
        super().__init__(bqm, seed)
        self.num_reads = num_reads
        self.name = "BQM_Dimod_SimulatedAnnealing"

    def solve(self) -> BQMSolutionResult:
        start_time = time.time()
        sampler = SimulatedAnnealingSampler()

        sampleset = sampler.sample(self.bqm, num_reads=self.num_reads, seed=self.seed if hasattr(self, 'seed') else None)
        best = sampleset.first

        best_sample = best.sample
        best_energy = best.energy

        elapsed_time = time.time() - start_time
        return BQMSolutionResult(
            method_name=self.name,
            energy=best_energy,
            sample=best_sample,
            computation_time=elapsed_time,
            iterations=self.num_reads,
            num_variables=self.num_vars
        )

def costruisci_qubo(capitale, unita_capitale, opzioni_data, antisinergie_data, penalty_strength=1):
    """Build QUBO from input data"""
    K = unita_capitale
    num_opzioni = len(opzioni_data)
    
    # Build coefficient list
    coefficienti = [(opt.a, opt.b, opt.c) for opt in opzioni_data]
    opzioni_names = [opt.name for opt in opzioni_data]
    
    # Build antisynergy list
    antisinergie = [((anti.option1, anti.option2), anti.value) for anti in antisinergie_data]

    # Costruzione CQM
    cqm = ConstrainedQuadraticModel()
    x = {i: Integer(f"x_{i}", lower_bound=0, upper_bound=K) for i in range(len(opzioni_names))}

    # Funzione obiettivo
    objective = sum([
        -(a * x[i] + b * x[i]**2)
        for i, (a, b, c) in enumerate(coefficienti)
    ])
    
    # Aggiunta antisinergie
    for (op1, op2), valore in antisinergie:
        try:
            i = opzioni_names.index(op1)
            j = opzioni_names.index(op2)
            objective += valore * (x[i]) * (x[j])
        except ValueError:
            continue
    
    # Vincolo capitale
    capital_sum = sum(x[i] for i in range(len(opzioni_names)))
    excess = capital_sum - K
    objective += penalty_strength * (excess * excess)
    
    # Imposta obiettivo
    cqm.set_objective(objective)

    # Conversione in QUBO
    bqm, invert_map = cqm_to_bqm(cqm)
    
    return bqm, cqm

def decode_bqm_solution_to_integers(sample: dict, num_variables: int) -> list:
    """Convert BQM binary solution back to integer allocations"""
    integers = [0] * num_variables
    
    for key, bit_value in sample.items():
        if not bit_value:
            continue
        
        if isinstance(key, tuple) and len(key) >= 2:
            var_name, bit_weight = key[0], key[1]
            if var_name.startswith("x_") and isinstance(bit_weight, int):
                idx = int(var_name.split("_")[1])
                if idx < num_variables:
                    integers[idx] += int(bit_value) * bit_weight
        
        elif isinstance(key, str) and key.startswith("x_"):
            idx = int(key.split("_")[1])
            if idx < num_variables:
                integers[idx] += int(bit_value)
    
    return integers

def solve_bqm_comprehensive(bqm: BinaryQuadraticModel, seed: Optional[int] = None) -> List[BQMSolutionResult]:
    """Solve BQM using multiple methods"""
    
    # Initialize solvers (reduced set for server)
    solvers = [
        BQMSimulatedAnnealingSolver(bqm, initial_temp=10.0, final_temp=1e-3, max_iterations=1000, seed=seed),
        DimodAnnealingSolver(bqm, num_reads=100, seed=seed),
    ]
    
    results = []
    
    for solver in solvers:
        try:
            result = solver.solve()
            results.append(result)
        except Exception as e:
            print(f"ERROR in {solver.name}: {e}")
            continue
    
    # Sort by energy (lower is better)
    results.sort(key=lambda x: x.energy)
    
    return results

@app.post("/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    try:
        # Build QUBO
        bqm, cqm = costruisci_qubo(
            request.capitale, 
            request.unita_capitale, 
            request.opzioni,
            request.antisinergie,
            request.penalty_strength
        )
        
        # Get QUBO matrix
        Q, offset = bqm.to_qubo()
        variables = list(bqm.variables)
        
        # Build symmetric matrix
        n = len(variables)
        Qmat = np.zeros((n, n))
        for (u, v), bias in Q.items():
            i = variables.index(u)
            j = variables.index(v)
            Qmat[i, j] += bias
            if i != j:
                Qmat[j, i] += bias
        
        # Solve QUBO
        results = solve_bqm_comprehensive(bqm, seed=42)
        
        # Convert results to response format
        response_results = []
        for result in results:
            # Convert numpy/dict values to JSON serializable
            sample_serializable = {}
            for k, v in result.sample.items():
                sample_serializable[str(k)] = int(v)
            
            response_results.append(SolutionResult(
                method_name=result.method_name,
                energy=float(result.energy),
                sample=sample_serializable,
                computation_time=float(result.computation_time),
                iterations=int(result.iterations),
                num_variables=int(result.num_variables)
            ))
        
        if not results:
            raise HTTPException(status_code=500, detail="No solutions found")
        
        # Best solution details
        best_result = results[0]
        best_frazioni = decode_bqm_solution_to_integers(best_result.sample, len(request.opzioni))
        
        best_solution = {
            "method": best_result.method_name,
            "objective": float(best_result.energy),
            "allocations": best_frazioni,
            "capital_values": [fraz * request.unita_capitale for fraz in best_frazioni],
            "total_capital": sum(fraz * request.unita_capitale for fraz in best_frazioni)
        }
        
        return PortfolioResponse(
            results=response_results,
            best_solution=best_solution,
            qubo_matrix=Qmat.tolist(),
            qubo_variables=variables,
            qubo_offset=float(offset)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
