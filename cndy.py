"""
Author: Noeloikeau Charlot
Date: 3/18/2025
Version: 0.1

Complex Network Dynamics (CNDY) Simulation Framework.

This module provides a framework for simulating complex network dynamics with
support for conditional JIT compilation via Numba.
"""

import inspect
import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt


# ---------- GLOBALS & META-FUNCTIONS ----------

# Global JIT configuration
USE_JIT = True
PARALLEL = False
CACHE = False


def conditional_jit(func, cache=CACHE, parallel=PARALLEL):
    """
    A simple conditional JIT decorator based on global USE_JIT setting.
    
    Args:
        func (callable): The function to potentially JIT compile
        cache (bool): Whether to cache compiled function
        parallel (bool): Whether to parallelize execution when possible
        
    Returns:
        callable: Either the JIT-compiled function (if USE_JIT is True) 
                 or the original function
    """
    if USE_JIT:
        return njit(cache=cache, parallel=parallel)(func)
    return func


def dispatcher(functions, atleast_2d=True, cache=True):
    """
    Create a dispatcher meta-function that calls one of the provided functions based on an ID.
    
    Args:
        functions (dict or list or set): A collection of functions to dispatch to.
                If dict: Maps integers to callables
                If list/set: Callables assigned consecutive integer IDs starting from 0.
        atleast_2d (bool): Whether to ensure output is at least 2D
        cache (bool): Whether to cache compiled functions
    
    Returns:
        callable: A function that dispatches to the appropriate function based on the ID.
                The signature matches the original functions with an added 'id' parameter.
                The function has a .bindings attribute that maps functions to their IDs.
                
    Raises:
        TypeError: If functions is not a dict, list, or set
        ValueError: If functions is empty or function signatures don't match
    """
    # Standardize functions to a dictionary with integer keys
    if isinstance(functions, dict):
        func_dict = functions
    elif isinstance(functions, (list, set)):
        func_dict = {i: func for i, func in enumerate(functions)}
    else:
        raise TypeError("functions must be a dict, list, or set")
    
    # Check if functions is empty
    if not func_dict:
        raise ValueError("No functions provided")
    
    # Extract the signature from the first function
    first_func = next(iter(func_dict.values()))
    sig = inspect.signature(first_func)
    
    # Check that all functions have the same signature
    if len(func_dict) > 1:
        for func in func_dict.values():
            if inspect.signature(func) != sig:
                raise ValueError(
                    f"Function signatures do not match: {sig} vs {inspect.signature(func)}"
                )
            
    # Attempt to jit each function if JIT is enabled
    for id in list(func_dict):
        f0 = func_dict[id]
        if isinstance(f0, numba.core.registry.CPUDispatcher):
            pass
        else:
            if USE_JIT:
                try:
                    f = njit(cache=cache)(f0)
                    func_dict[id] = f
                except Exception:
                    raise ValueError(
                        f"Function {f0.__name__} could not be jitted. "
                        "It must be Numba compatible."
                    )
    
    # Create dispatcher function
    # Get parameter names from the signature
    param_names = list(sig.parameters.keys())
    param_str = ", ".join(param_names)

    # Generate name based on input functions
    func_names = [func.__name__ for func in func_dict.values()]
    dispatcher_name = "_".join(func_names) + "_dispatcher"
    
    # Generate the dispatcher function code with the original function's signature plus 'id'
    func_code = ""
    
    # Conditionally add the njit decorator based on USE_JIT
    if USE_JIT:
        func_code += "@njit\n"
    
    func_code += f"""def {dispatcher_name}({param_str}, id: int = 0):
    '''
    Dispatch to a function based on the ID.
    Casts input to numpy array and output to at least 2D for numba compatibility.
    
    Args:
        id: The ID of the function to dispatch to.
        [Original function parameters]
    
    Returns:
        The result of applying the function to the inputs.
    '''
"""
    
    # Generate if-elif blocks for each function
    for i, (id, func) in enumerate(func_dict.items()):
        func_name = func.__name__
        if i == 0:
            func_code += f"    if id == {id}:\n"
        else:
            func_code += f"    elif id == {id}:\n"
        if atleast_2d:
            func_code += f"        result = np.atleast_2d(np.asarray({func_name}({param_str})))\n"
        else:
            func_code += f"        result = {func_name}({param_str})\n"
    
    func_code += "    return result\n"
    
    # Compile the function code
    global_vars = {
        'np': np,
        'njit': njit,
    }
    # Add the functions to the global variables
    for func in func_dict.values():
        global_vars[func.__name__] = func
    
    local_vars = {}
    exec(func_code, global_vars, local_vars)
    dispatcher_func = local_vars[dispatcher_name]
            
    # Add dispatcher bindings
    dispatcher_func.bindings = {id: func for id, func in func_dict.items()}
    
    return dispatcher_func


# ---------- INITIAL CONDITION ----------

def create_initial_history(initial_state, T, max_delay_value=None):
    """
    Create a history array specifying initial state(s) for network dynamics functions.
    
    Args:
        initial_state (np.ndarray): Initial state of the network [N, M]
        T (int): Number of time steps to simulate
        max_delay_value (float, optional): Maximum delay in the network. Defaults to None.
        
    Returns:
        np.ndarray: Initial network history [T, N, M]
    
    Notes:
        Pads the back of the array with the initial state. This is because t - delay
        can be negative at early times, and the initial condition is accessed on the
        end of the array. The padding is then overwritten by net_dynamics as the 
        state evolves.
    """
    N, M = initial_state.shape
    history = np.zeros((T, N, M))

    # Copy initial state to the first time step
    history[0] = initial_state 

    # If max_delay_value is provided, treat negative t-delay as initial trajectory
    if max_delay_value:
        # Pad up to max-1 so negative t-delay can access initial state
        for t in range(0, -round(max_delay_value)-1, -1):
            history[t] = initial_state
        
    return history


# ---------- NODE FUNCTIONS ----------
# Nodes execute functions that have signatures of the form:
# node_function(node_state, node_input, node_data, time=0) -> node_state

@conditional_jit
def negation(state, input, data, time=0):
    """
    Negates the input to a single node.
    
    Args:
        state: Current state [M]
        input: Node inputs [K, M]
        data: Node parameters
        time: Current time step
        
    Returns:
        New state after applying negation [M]
    """
    input_sum = np.sum(input)
    return 1.0 - input_sum


@conditional_jit
def summation(state, input, data, time=0):
    """
    Computes state + sum(input).
    
    Args:
        state: Current state [M]
        input: Node inputs [K, M]
        data: Node parameters
        time: Current time step
        
    Returns:
        New state after applying summation [M]
    """
    input_sum = np.sum(input)
    return state + input_sum


@conditional_jit
def subtraction(state, input, data, time=0):
    """
    Computes state - sum(input).
    
    Args:
        state: Current state [M]
        input: Node inputs [K, M]
        data: Node parameters
        time: Current time step
        
    Returns:
        New state after applying subtraction [M]
    """
    input_sum = np.sum(input)
    return state - input_sum


@conditional_jit
def spiking(node_state, node_input, node_params, time=0):
    """
    Implements a spiking neuron model.
    
    Args:
        node_state: Current state [M]
        node_input: Node inputs [K, M]
        node_params: Node parameters [bias, threshold]
        time: Current time step
        
    Returns:
        New state after applying spiking dynamics [M]
    """
    bias, threshold = node_params[0], node_params[1]
    weighted_spikes = node_input[:, 1]
    current = threshold * (weighted_spikes.sum() + bias)
    potential = node_state[0] + current
    spike = 1.0 if potential >= threshold else 0.0
    potential -= spike * threshold
    return np.array([potential, spike])


# ---------- EDGE FUNCTIONS ----------
# Edges execute functions that have signatures of the form:
# edge_function(source, target, net_history, time=0) -> node_input

@conditional_jit
def waited(source, target, state, data, time=0):
    """
    Delays a signal: weight * source_state(time-delay).
    
    Args:
        source: Source node index
        target: Target node index
        state: Network state history [T, N, M]
        data: Edge parameters [weight, delay]
        time: Current time step
        
    Returns:
        Delayed signal from source to target
    """
    weight, delay = data
    source_state = state[int(time - delay), int(source)]
    return weight * source_state


@conditional_jit
def kicked(source, target, state, data, time=0):
    """
    Applies a kick to a signal: weight * (source_state(time-delay) + kick),
    where kick alternates between +1 and -1 based on time.
    
    Args:
        source: Source node index
        target: Target node index
        state: Network state history [T, N, M]
        data: Edge parameters [weight, delay]
        time: Current time step
        
    Returns:
        Kicked signal from source to target
    """
    weight, delay = data
    kick_value = -1 if time % 2 == 0 else 1
    source_state = state[int(time - delay), int(source)]
    return weight * (source_state + kick_value)


# ---------- NETWORK FUNCTIONS ----------
# Network functions can modify node and edge data. They have signatures of the form: 
# network_function(node_data, edge_data, history, time=0) -> node_data, edge_data

@conditional_jit
def static_network(node_data, edge_data, history, time=0.0):
    """
    A static network function that does not modify node or edge data.
    
    Args:
        node_data: Node data array
        edge_data: Edge data array
        history: Network state history
        time: Current time step
        
    Returns:
        Unchanged node and edge data
    """
    return node_data, edge_data


# ---------- NETWORK INTEGRATOR ----------
# Integrate the network dynamics by searching all nodes and edges at every time step. 
# This is the simplest, most general approach. We use numba for loop optimization.

@conditional_jit
def conditional_range(N):
    """
    Returns a range or prange based on JIT configuration.
    
    Args:
        N: Size of the range
        
    Returns:
        Range or prange based on configuration
    """
    if USE_JIT and PARALLEL:
        return numba.prange(N)
    else:
        return range(N)


@conditional_jit
def network_dynamics(
    net_history, 
    node_data, 
    edge_data, 
    node_dispatcher, 
    edge_dispatcher,
    time=0.0, 
    dt=1.0, 
    network_function=static_network
):
    """
    Integrates the network dynamics over time using brute force search with numba compatibility.
    
    Args:
        net_history: Network state history [T, N, M]
        node_data: Node data array
        edge_data: Edge data array
        node_dispatcher: Node function dispatcher
        edge_dispatcher: Edge function dispatcher
        time: Initial time
        dt: Time step
        network_function: Function to modify network structure over time
        
    Returns:
        Updated network state history
    """
    # Copy input and shape
    net_history = net_history.copy()
    T, N, M = net_history.shape
    
    # Integrate over time steps (skip t=0 which is initial condition)
    for t in range(1, T):
        # Update temporal variables
        current_time = time + t * dt
        previous_time = time + (t-1) * dt
        node_data, edge_data = network_function(node_data, edge_data, net_history, current_time)
        
        # Count incoming edges for all nodes to pre-allocate array
        in_degree = np.zeros(N, dtype=np.int32)
        for edge_index in range(len(edge_data)):
            target_node = int(edge_data[edge_index, 1])
            in_degree[target_node] += 1
            
        for n in conditional_range(N):
            # Get node's previous state and initialize inputs
            previous_state = net_history[t-1, n]
            node_inputs = np.zeros((in_degree[n], M))
            
            # Track current position in node_inputs array
            current_input = 0
            
            # Process all edges to find inputs to this node
            for edge_index in range(len(edge_data)):
                source_node = int(edge_data[edge_index, 0])
                target_node = int(edge_data[edge_index, 1])
                edge_function_id = int(edge_data[edge_index, 2])
                edge_params = edge_data[edge_index, 3:]
                
                if target_node == n:
                    # Apply edge function
                    res = edge_dispatcher(
                        source_node, 
                        n, 
                        net_history, 
                        edge_params, 
                        previous_time, 
                        edge_function_id
                    )
                    
                    # Store result in pre-allocated array
                    node_inputs[current_input] = res
                    current_input += 1
            
            # Get node's function ID and parameters
            node_function_id = int(node_data[n, 1])
            node_params = node_data[n, 2:]
            
            # Apply node dynamics to update state
            current_state = node_dispatcher(
                previous_state, 
                node_inputs, 
                node_params, 
                previous_time, 
                node_function_id
            )
            
            # Update network history
            net_history[t, n] = current_state
    
    return net_history


# ---------- VISUALIZATION ----------

def plot_net(
    result, 
    node_data_array, 
    edge_data_array, 
    node_dispatcher, 
    edge_dispatcher, 
    figsize=(10, 5), 
    dark_background=True, 
    use_grid=True, 
    cm='hsv',
    text_fontsize=9, 
    linewidth=2, 
    text_offset=1.02, 
    subplot_adjust_right=0.7,
    max_text_lines=20
):
    """
    Plot network dynamics with color-coded node and edge labels.
    
    Parameters:
    ----------
    result : ndarray
        Simulation results with shape (time_steps, num_nodes, state_dimension)
    node_data_array : ndarray
        Node data including node IDs and functions
    edge_data_array : ndarray
        Edge data including source, target, edge functions, and parameters
    node_dispatcher : object
        Node function dispatcher with bindings property
    edge_dispatcher : object
        Edge function dispatcher with bindings property
    figsize : tuple, optional
        Figure size (width, height) in inches
    dark_background : bool, optional
        Whether to use dark background style
    use_grid : bool, optional
        Whether to show grid lines
    cm : str, optional
        Colormap name
    text_fontsize : int, optional
        Font size for text elements
    linewidth : int, optional
        Line width for plotted lines
    text_offset : float, optional
        Starting x position for text labels (as a fraction of axes width)
    subplot_adjust_right : float, optional
        Right margin adjustment to make room for labels
    max_text_lines : int, optional
        Maximum number of text lines to display on the right side
    
    Returns:
    -------
    fig : matplotlib Figure
        The created figure
    ax : matplotlib Axes
        The created axes
    """
    # Network parameters
    N = node_data_array.shape[0]  # Number of nodes
    M = result.shape[2] if len(result.shape) > 2 else 1  # State dimension per node
    T = result.shape[0]  # Number of time steps
    
    # Set plot style
    if dark_background:
        plt.style.use('dark_background')
    
    figs = []
    axes = []
    
    # Organize all incoming edges by target node
    node_edges = {i: [] for i in range(N)}
    for e in edge_data_array:
        target = int(e[1])
        node_edges[target].append(e)
    
    # Calculate total lines needed (1 line per node equation + 1 line per edge)
    total_lines = N
    total_edges = 0
    for i in range(N):
        total_edges += len(node_edges[i])
    total_lines += total_edges
    
    # Determine if we need to sample
    sampling_needed = total_lines > max_text_lines
    if sampling_needed:
        # Calculate what fraction of nodes and edges to include
        sample_fraction = max_text_lines / total_lines
        nodes_to_show = max(3, int(N * sample_fraction))  # Show at least 3 nodes
        
        # Always include first, middle, and last nodes
        important_nodes = [0, N//2, N-1]
        important_nodes = list(set(important_nodes))  # Remove duplicates (in case N is small)
        
        # If we have more slots to fill, use uniform sampling across the range
        if nodes_to_show > len(important_nodes):
            # Create a set of candidate nodes (excluding already selected ones)
            candidates = [i for i in range(N) if i not in important_nodes]
            remaining_slots = nodes_to_show - len(important_nodes)
            
            if candidates:
                if len(candidates) <= remaining_slots:
                    # If we can show all candidates, include them all
                    important_nodes.extend(candidates)
                else:
                    # Calculate step size for uniform sampling
                    step = (N - 1) / (nodes_to_show - 1)
                    
                    # Generate uniformly spaced indices
                    uniform_indices = set()
                    for i in range(nodes_to_show):
                        idx = min(round(i * step), N-1)
                        uniform_indices.add(idx)
                    
                    # Remove already included indices and convert to list
                    uniform_indices = [
                        idx for idx in uniform_indices 
                        if idx not in important_nodes
                    ]
                    
                    # Add as many as we need
                    important_nodes.extend(uniform_indices[:remaining_slots])
        
        # Sort for better readability
        important_nodes.sort()
    else:
        important_nodes = list(range(N))
    
    for m in range(M):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color dictionary for each node
        node_colors = {i: getattr(plt.cm, cm)(i/N) for i in range(N)}
        
        # Plot the lines for all nodes (even if we don't show their equations)
        for i in range(N):
            ax.plot(
                result[:, i, m] if M > 1 else result[:, i], 
                color=node_colors[i], 
                lw=linewidth
            )
        
        # Starting position for text elements
        y_pos = 0.95
        x_start = text_offset
        
        # Process node equations and their edges
        for i in important_nodes:
            # Find incoming neighbors for this node
            neighbors = []
            for e in edge_data_array:
                if e[1] == i:
                    neighbors.append(int(e[0]))
            
            # Node function name
            node_fn_name = node_dispatcher.bindings[node_data_array[i, 1]].__name__
            
            # Create the node equation as separate text elements with appropriate colors
            x_pos = x_start
            
            # Left side of equation (xi =)
            ax.text(
                x_pos, 
                y_pos, 
                "x", 
                transform=ax.transAxes, 
                fontsize=text_fontsize, 
                verticalalignment='top', 
                color=node_colors[i]
            )
            x_pos += 0.015
            
            ax.text(
                x_pos, 
                y_pos, 
                f"{i}", 
                transform=ax.transAxes, 
                fontsize=text_fontsize, 
                verticalalignment='top', 
                color=node_colors[i]
            )
            x_pos += 0.015
            
            ax.text(
                x_pos, 
                y_pos, 
                " = ", 
                transform=ax.transAxes, 
                fontsize=text_fontsize, 
                verticalalignment='top', 
                color="white"
            )
            x_pos += 0.04
            
            # Function name with extra spacing
            ax.text(
                x_pos, 
                y_pos, 
                f"{node_fn_name}( ", 
                transform=ax.transAxes, 
                fontsize=text_fontsize, 
                verticalalignment='top', 
                color="white"
            )
            x_pos += len(node_fn_name) * 0.01 + 0.03  # Extra space after function name
            
            # Add neighbors with their respective colors
            for j, neighbor in enumerate(neighbors):
                ax.text(
                    x_pos, 
                    y_pos, 
                    "x", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color=node_colors[neighbor]
                )
                x_pos += 0.015
                
                ax.text(
                    x_pos, 
                    y_pos, 
                    f"{neighbor}", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color=node_colors[neighbor]
                )
                x_pos += 0.015
                
                if j < len(neighbors) - 1:
                    ax.text(
                        x_pos, 
                        y_pos, 
                        ", ", 
                        transform=ax.transAxes, 
                        fontsize=text_fontsize, 
                        verticalalignment='top', 
                        color="white"
                    )
                    x_pos += 0.03
            
            # Close parenthesis with extra space
            ax.text(
                x_pos, 
                y_pos, 
                " )", 
                transform=ax.transAxes, 
                fontsize=text_fontsize, 
                verticalalignment='top', 
                color="white"
            )
            
            y_pos -= 0.04
            
            # Process edge equations for this node
            if sampling_needed and len(node_edges[i]) > 0:
                # If there are many edges for this node, sample them
                edges_to_show = min(
                    len(node_edges[i]), 
                    max(1, int(max_text_lines / len(important_nodes)))
                )
                edge_sample = node_edges[i][:edges_to_show]
                
                # If we're sampling, add a note about how many edges are not shown
                if len(edge_sample) < len(node_edges[i]):
                    x_pos = x_start
                    ax.text(
                        x_pos, 
                        y_pos, 
                        f"... and {len(node_edges[i]) - len(edge_sample)} more edges", 
                        transform=ax.transAxes, 
                        fontsize=text_fontsize,
                        verticalalignment='top', 
                        color="gray", 
                        alpha=0.7
                    )
                    y_pos -= 0.04
            else:
                edge_sample = node_edges[i]
            
            for e in edge_sample:
                source = int(e[0])
                edge_fn = edge_dispatcher.bindings[e[2]].__name__
                params = ', '.join([str(p) for p in e[3:]])
                
                x_pos = x_start  # No indentation
                
                # Target node
                ax.text(
                    x_pos, 
                    y_pos, 
                    "x", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color=node_colors[i]
                )
                x_pos += 0.015
                
                ax.text(
                    x_pos, 
                    y_pos, 
                    f"{i}", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color=node_colors[i]
                )
                x_pos += 0.015
                
                # Use prettier Unicode arrow
                ax.text(
                    x_pos, 
                    y_pos, 
                    " ← ", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color="white"
                )
                x_pos += 0.04
                
                # Source node
                ax.text(
                    x_pos, 
                    y_pos, 
                    "x", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color=node_colors[source]
                )
                x_pos += 0.015
                
                ax.text(
                    x_pos, 
                    y_pos, 
                    f"{source}", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color=node_colors[source]
                )
                x_pos += 0.015
                
                ax.text(
                    x_pos, 
                    y_pos, 
                    " : ", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color="white"
                )
                x_pos += 0.025
                
                # Edge function with parameters
                ax.text(
                    x_pos, 
                    y_pos, 
                    f"{edge_fn}({params})", 
                    transform=ax.transAxes, 
                    fontsize=text_fontsize, 
                    verticalalignment='top', 
                    color="white"
                )
                
                y_pos -= 0.04
            
            # Add extra space between nodes
            if node_edges[i]:  # If there were edges for this node
                y_pos -= 0.02  # Add extra spacing
        
        # If we sampled nodes, add info about how many nodes are not shown
        if len(important_nodes) < N:
            x_pos = x_start
            ax.text(
                x_pos, 
                y_pos, 
                f"... showing {len(important_nodes)}/{N} nodes", 
                transform=ax.transAxes, 
                fontsize=text_fontsize,
                verticalalignment='top', 
                color="gray", 
                alpha=0.7
            )
            y_pos -= 0.04
        
        plt.xlabel('Time')
        if T <= 20:
            plt.xticks(range(T))
        plt.ylabel(f'Node State (Axis {m})')
        plt.title('Complex Network Dynamics (CNDY) Simulation')
        
        if use_grid:
            plt.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(right=subplot_adjust_right)  # Make room for the labels
        
        figs.append(fig)
        axes.append(ax)
    
    if M == 1:
        return figs[0], axes[0]
    else:
        return figs, axes


# ---------- TEST SIMULATIONS ----------

def simple_test(cm='hsv'):
    """
    Run a simple test of the network dynamics framework.
    
    Args:
        cm (str, optional): Colormap name for plotting. Defaults to 'hsv'.
        
    Returns:
        dict: Local variables from test for inspection
    """
    # Numba-compliant dispatchers require pre-allocation of output array dimensions
    atleast_2d = True

    # single-input dispatchers for naive implementation
    node_dispatcher = dispatcher({0: negation, 1: summation, 2: subtraction}, atleast_2d=atleast_2d)
    edge_dispatcher = dispatcher({0: waited, 1: kicked}, atleast_2d=atleast_2d)

    # edge data: [source, target, edge function (, edge parameters)]
    edge_data_array = np.array([
        # source, target, function ID, weight, delay
        [0, 1, 0, 1.0, 1],  # Node 0 -> Node 1, edge function 0, weight 1.0, delay 1
        [1, 2, 0, 1.0, 0],  # Node 1 -> Node 2, edge function 0, weight 1.0, delay 0
        [2, 0, 0, 1.0, 0],  # Node 2 -> Node 0, edge function 0, weight 1.0, delay 0
        [2, 3, 0, 0.5, 0],  # Node 2 -> Node 3, edge function 0, weight 0.5, delay 0
        [4, 4, 1, 0.25, 0],  # Node 4 -> Node 4, edge function 1, weight 0.25, delay 0
        [4, 4, 0, 0.25, 0],  # Node 4 -> Node 4, edge function 0, weight 0.25, delay 0
        [5, 3, 1, 0.5, 0],   # Node 5 -> Node 3, edge function 1, weight 0.5, delay 0
        [5, 3, 1, 0.5, 0],   # Node 5 -> Node 3, edge function 1, weight 0.5, delay 0
    ])

    # node data: [node ID, node function (, node parameters)]
    node_data_array = np.array([
        [0, 0],  # node 0, node function 0
        [1, 0],  # node 1, node function 0
        [2, 0],  # node 2, node function 0
        [3, 1],  # node 3, node function 1
        [4, 2],  # node 4, node function 2
        [5, 1],  # node 5, node function 1
    ])
    
    # Network parameters
    N = node_data_array.shape[0]  # Number of nodes
    M = 1  # State dimension per node
    T = 10  # Number of time steps
    max_delay_value = edge_data_array[:, -1].max()
    
    # Initial state
    initial_state = np.array([[1], [0], [0], [0], [1 + edge_data_array[-1, -2]], [0]])
    initial_history = create_initial_history(initial_state, T, max_delay_value)
    
    # Run simulations
    result = network_dynamics(
        initial_history, 
        node_data_array, 
        edge_data_array, 
        node_dispatcher, 
        edge_dispatcher
    )
    
    # Plot network dynamics
    fig, ax = plot_net(
        result, 
        node_data_array, 
        edge_data_array, 
        node_dispatcher, 
        edge_dispatcher, 
        cm=cm
    )
    
    # Return local variables for inspection
    return locals()


def spiking_test(
    b=0.0,       # bias
    h=0.5,       # threshold
    w=0.2,       # weight
    d=0.0,       # delay
    D=0.0,       # big delay (node 5)
    W=1.0,       # big weight (node 5)
    x1=-1,       # start of sinc input
    x2=20,       # end of sinc input
    T=100,       # time steps    
    m=1,         # state to plot, 1 is spike
    M=2,         # state dimension
    normalize=True                       
):
    """
    Run a test of a feedforward spiking neural network with a central recurrent node.
    
    Args:
        b (float): Bias for spiking nodes
        h (float): Threshold for spiking nodes
        w (float): Weight for standard connections
        d (float): Delay for standard connections (fraction of T)
        D (float): Delay for center node connections (fraction of T)
        W (float): Weight for center node connections
        x1 (float): Start of sinc input range
        x2 (float): End of sinc input range
        T (int): Number of time steps to simulate
        m (int): State component to plot (1=spike)
        M (int): State dimension per node
        normalize (bool): Whether to normalize sinc input
        
    Returns:
        np.ndarray: Result of the network simulation [T, N, M]
    
    Notes:
        Network topology:
        ```
             -> n1 -> n4 -> n7 ->
                |   \  | /  |
          n0 -> n2 -> n5 -> n8 -> n10
                |   /  | \  |
             -> n3 -> n6 -> n9 ->
        ```
        
        - Node 0 is input - implements sinc(x[t])
        - Everything else is a spiking node
        - Node 5 in center has biggest weight/delay
        - [1,2,3], [4,6], [7,8,9] are feedforward layers
    """
    # Sinc parameters and definition
    x = np.linspace(x1, x2, T)
    y = np.sinc(x)
    Y = (y - y.min()) / (y.max() - y.min())
    
    @conditional_jit
    def sinc(node_state, node_input, node_params, time=0.0): 
        """
        Sinc function node.
        
        Args:
            node_state: Current state [M]
            node_input: Node inputs [K, M]
            node_params: Node parameters
            time: Current time step
            
        Returns:
            Sinc value at current time step
        """
        time = int(time)
        z = y[time] if not normalize else Y[time]
        # degenerate source state (spike=voltage)
        return np.array([[z] * M])
        
    # dispatch - {id: function}
    node_funcs = {0: sinc, 1: spiking}
    edge_funcs = {0: waited}
    node_dispatcher = dispatcher(node_funcs, atleast_2d=True)
    edge_dispatcher = dispatcher(edge_funcs, atleast_2d=True)
    
    # All nodes have same function except node 0
    node_data = np.array([[0, 0, b, h]] + [[i, 1, b, h] for i in range(1, 11)])
    N = node_data.shape[0]
    
    edge_data = [
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 4], [1, 5],
        [2, 1], [2, 3], [2, 5],
        [3, 2], [3, 5], [3, 6],
        [4, 5], [4, 7],
        [5, 1], [5, 3], [5, 4], [5, 6], [5, 7], [5, 9],
        [6, 5], [6, 9],
        [7, 8],
        [9, 8],
        [7, 10], [8, 10], [9, 10]
    ]
    
    # All edges have same parameters except edges flowing out of node 5
    edge_data = np.array(
        [
            [
                u, v, 0,
                w if u != 5 else W,
                round(d * T) if u != 5 else round(D * T)
            ] 
            for u, v in edge_data
        ], 
        dtype=float
    )

    # Initialize all time to zero for impulse response
    initial_history = np.zeros((T, N, M))
    result = network_dynamics(
        initial_history, 
        node_data, 
        edge_data, 
        node_dispatcher, 
        edge_dispatcher
    )

    fig, ax = plot_net(
        result, 
        node_data, 
        edge_data, 
        node_dispatcher, 
        edge_dispatcher
    )

    return result


# Run tests if this module is executed as a script
if __name__ == "__main__":
    print("Running simple test...")
    simple_test_result = simple_test()
    
    print("Running spiking test...")
    spiking_test_result = spiking_test()
    
    plt.show()
