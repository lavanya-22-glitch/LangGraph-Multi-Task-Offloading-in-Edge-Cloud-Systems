`
workflow_dict = {
"tasks": {
1: {"v": 4e6}, # 4M cycles (light)
2: {"v": 20e6}, # 20M cycles (heavy)
3: {"v": 6e6}, # 6M cycles (medium)
4: {"v": 12e6}, # 12M cycles (heavy)
5: {"v": 5e6}, # 5M cycles (light, sink)
},
"edges": {
(1, 3): 5e6, # 5 MB
(2, 3): 12e6, # 12 MB (big!)
(3, 4): 2e6, # 2 MB
(3, 5): 6e6, # 6 MB
(4, 5): 1e6, # 1 MB (join at task 5)
},
"N": 5
}

    # Create Workflow object from experiment dict
    wf = Workflow.from_experiment_dict(workflow_dict)

    # ------------------------ ENVIRONMENT DEFINITION ------------------------
    # Define location types: 0=IoT (mandatory), 1+=edge/cloud
    locations_types = {
    0: "iot",
    1: "edge",
    2: "edge",
    3: "cloud",
    4: "cloud",

}

    # DR: Data Time Consumption (ms/byte) - time to transfer 1 byte between locations
    DR_map = {
    (0,0):0.0,      (1,1):0.0,      (2,2):0.0,      (3,3):0.0,      (4,4):0.0,

    # IoT <-> Edges
    (0,1):0.00012,  (1,0):0.00012,  # 0.12 ms/MB
    (0,2):0.00018,  (2,0):0.00018,  # 0.18 ms/MB

    # IoT <-> Clouds (CloudB slower uplink from IoT)
    (0,3):0.00150,  (3,0):0.00150,  # 1.5 ms/MB
    (0,4):0.00250,  (4,0):0.00250,  # 2.5 ms/MB

    # Edge <-> Edge
    (1,2):0.00008,  (2,1):0.00008,  # 0.08 ms/MB

    # Edge <-> Clouds
    (1,3):0.00050,  (3,1):0.00050,  # 0.5 ms/MB
    (1,4):0.00100,  (4,1):0.00100,  # 1.0 ms/MB
    (2,3):0.00060,  (3,2):0.00060,  # 0.6 ms/MB
    (2,4):0.00070,  (4,2):0.00070,  # 0.7 ms/MB

    # Cloud <-> Cloud (very fast)
    (3,4):0.00005,  (4,3):0.00005,  # 0.05 ms/MB

}

    # DE: Data Energy Consumption (mJ/byte) - energy to process 1 byte at location
    DE_map = {
    0: 0.00012,   # IoT
    1: 0.00006,   # EdgeA
    2: 0.00005,   # EdgeB (slightly better than EdgeA)
    3: 0.00003,   # CloudA
    4: 0.00002,   # CloudB (best)

}

    # VR: Task Time Consumption (ms/cycle) - time to execute 1 CPU cycle
    VR_map = {
    0: 1.2e-7,    # IoT (slowest)
    1: 3.0e-8,    # EdgeA
    2: 2.2e-8,    # EdgeB
    3: 1.4e-8,    # CloudA
    4: 1.0e-8,    # CloudB (fastest)

}

    # VE: Task Energy Consumption (mJ/cycle) - energy per CPU cycle
    VE_map = {
    0: 6.0e-7,    # IoT (least efficient)
    1: 2.5e-7,    # EdgeA
    2: 2.0e-7,    # EdgeB
    3: 1.4e-7,    # CloudA
    4: 1.1e-7,    # CloudB (most efficient)

}

    # Create environment dictionary
    env_dict = create_environment_dict(
        locations_types=locations_types,
        DR_map=DR_map,
        DE_map=DE_map,
        VR_map=VR_map,
        VE_map=VE_map
    )

    # Create Environment object
    env = Environment.from_matrices(
        types=locations_types,
        DR_matrix=DR_map,
        DE_vector=DE_map,
        VR_vector=VR_map,
        VE_vector=VE_map
    )

    # ------------------------ OPTIMIZATION PARAMETERS -----------------------
    # Cost coefficients and mode as per the paper
    params = {
        "CT": 0.18,      # Cost per unit time (Eq. 1)
        "CE": 1.20,     # Cost per unit energy (Eq. 2)
        "delta_t": 1,   # Weight for time cost (1=enabled, 0=disabled)
        "delta_e": 1,   # Weight for energy cost (1=enabled, 0=disabled)
    }

`
