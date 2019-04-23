# AnalysisOfViableParameterRegions

A computational approach for glocal analysis of viable parameter regions in biological gene regulatory network models. The methodology is based on the exploration of high-dimensional viable parameter spaces with genetic algorithms and local sampling in the direction of principal components. This methodology can be applied to models with poorly connected viable solution spaces.   

## Getting Started

The main python file [solver.py](https://github.com/zigapusnik/AnalysisOfViableParameterRegions/blob/master/solver.py) consists of two classes: Solver and Region. When instantiating a solver object, GRN model must be specified. See for example [D flip-flop model](https://github.com/zigapusnik/AnalysisOfViableParameterRegions/blob/master/dFlipFlop.py).  

### Prerequisites

The source code is written in

* python 3.7 with [deap](https://deap.readthedocs.io/en/master/) and [scikit-learn](https://scikit-learn.org/stable/).

## Running the tests

When running experiments you need to instantiate the model and provide a path to folder where the results will be saved.
The experiments can be run with the following commands:

```
	path =  full_path    
	model = Model(parameter_values, parameter_names, initial_conditions)
	solver = Solver(model)         
	solver.run(path)      
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details






