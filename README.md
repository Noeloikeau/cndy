# CNDY: Complex Network Dynamics

## Overview

CNDY is a Python framework for simulating and analyzing dynamical complex networks with support for conditional Just-In-Time (JIT) compilation via Numba for high-performance computing. It supports:

- Configurable node and edge functions
- Time delays in signal propagation
- Dynamic network topology changes
- Visualization of network dynamics
- Efficient computation with Numba JIT compilation

## Key Components

### 1. Meta-Functions

- `conditional_jit`: A decorator that conditionally applies Numba's JIT compilation
- `dispatcher`: Creates a dispatcher function that selects between multiple functions based on an ID

### 2. Node Functions

Node functions process incoming signals and update node state:

- `negation`: Negates the sum of inputs
- `summation`: Adds the sum of inputs to the current state
- `subtraction`: Subtracts the sum of inputs from the current state
- `spiking`: Implements a leaky integrate-and-fire neuron model

### 3. Edge Functions

Edge functions transform signals between nodes:

- `waited`: Delays a signal with a weight
- `kicked`: Applies an alternating kick to a signal and delays it

### 4. Network Integrator

- `network_dynamics`: Integrates network dynamics over time using a brute force search with Numba compatibility

### 5. Visualization

- `plot_net`: Visualizes network dynamics with color-coded node and edge labels

## Example Usage

### Simple Network Test

```python
from cndy import simple_test

# Run a simple test with default parameters
result = simple_test()
```

This creates a small network with various node types and displays the dynamics.

### Spiking Neural Network Test

```python
from cndy import spiking_test

# Run a spiking neural network test
result = spiking_test(
    b=0.0,       # bias
    h=0.5,       # threshold
    w=0.2,       # weight
    d=0.1,       # delay
    normalize=True
)
```

This simulates a feedforward spiking neural network with the following topology:

```
     -> n1 -> n4 -> n7 ->
        |   \  | /  |
  n0 -> n2 -> n5 -> n8 -> n10
        |   /  | \  |
     -> n3 -> n6 -> n9 ->
```

## Customization

### Creating Custom Node Functions

```python
@conditional_jit
def custom_node_function(state, input, data, time=0):
    """
    Custom node function implementation.
    
    Args:
        state: Current state [M]
        input: Node inputs [K, M]
        data: Node parameters
        time: Current time step
        
    Returns:
        New state after applying the function [M]
    """
    # Your custom logic here
    return new_state
```

### Creating Custom Edge Functions

```python
@conditional_jit
def custom_edge_function(source, target, state, data, time=0):
    """
    Custom edge function implementation.
    
    Args:
        source: Source node index
        target: Target node index
        state: Network state history [T, N, M]
        data: Edge parameters
        time: Current time step
        
    Returns:
        Transformed signal from source to target
    """
    # Your custom logic here
    return transformed_signal
```

### Creating a Custom Network

```python
# Create node and edge function dispatchers
node_dispatcher = dispatcher({0: function1, 1: function2})
edge_dispatcher = dispatcher({0: edge_function1, 1: edge_function2})

# Define edge data: [source, target, edge function ID, parameters...]
edge_data = np.array([
    [0, 1, 0, 1.0, 1],  # Node 0 -> Node 1, edge function 0, weight 1.0, delay 1
    [1, 2, 1, 0.5, 0],  # Node 1 -> Node 2, edge function 1, weight 0.5, delay 0
])

# Define node data: [node ID, node function ID, parameters...]
node_data = np.array([
    [0, 0, param1, param2],  # node 0, function 0, parameters
    [1, 1, param1, param2],  # node 1, function 1, parameters
    [2, 0, param1, param2],  # node 2, function 0, parameters
])

# Create initial state and history
initial_state = np.array([[1], [0], [0]])  # [N, M]
T = 100  # Number of time steps
initial_history = create_initial_history(initial_state, T)

# Run simulation
result = network_dynamics(
    initial_history, 
    node_data, 
    edge_data, 
    node_dispatcher, 
    edge_dispatcher
)

# Visualize results
plot_net(result, node_data, edge_data, node_dispatcher, edge_dispatcher)
```

## Performance Considerations

### JIT Compilation Settings

The global settings for JIT compilation can be modified:

```python
USE_JIT = True    # Set to False to disable JIT compilation
PARALLEL = True   # Set to True to enable parallel execution
CACHE = True      # Set to True to cache compiled functions
```

### Large Networks

For large networks:

1. Consider using sampling in the visualization with `max_text_lines`
2. Adjust the plotting parameters to maintain readability
3. Use parallel execution with `PARALLEL = True` when possible

## Advanced Features

### Network Function for Dynamic Topology

You can create a custom network function to modify the network topology over time:

```python
@conditional_jit
def dynamic_network(node_data, edge_data, history, time=0.):
    """
    Dynamically modify network structure based on time or state.
    """
    # Modify node_data or edge_data based on time or state
    
    return modified_node_data, modified_edge_data
```

Then pass it to the network dynamics function:

```python
result = network_dynamics(
    initial_history, 
    node_data, 
    edge_data, 
    node_dispatcher, 
    edge_dispatcher,
    network_function=dynamic_network
)
```
