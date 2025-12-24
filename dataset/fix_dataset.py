import json

# --- REALISTIC PHYSICS SETTINGS ---
# 1. Processing Speed (Lower is Faster)
VR_SETTINGS = {
    "iot":   1.2e-7,   # Slow Base
    "edge":  1.2e-8,   # 10x Faster
    "cloud": 1.2e-9    # 100x Faster
}

# 2. Transmission Energy (IoT is expensive)
DE_SETTINGS = {
    "iot":   0.005,    # High Battery Drain
    "edge":  0.00001,  # Grid Power (Cheap)
    "cloud": 0.00001   # Grid Power (Cheap)
}

# 3. Network Latency (Cloud is far)
DR_SETTINGS = {
    "iot_to_edge": 0.0001,  # Fast (Wi-Fi)
    "iot_to_cloud": 0.0020  # Slow (WAN - 20x Penalty)
}

def fix_dataset(input_file='dataset.json', output_file='dataset_fixed.json'):
    with open(input_file, 'r') as f:
        dataset = json.load(f)

    print(f"Processing {len(dataset)} cases...")

    for case in dataset:
        loc_types = case['location_types']
        
        # --- FIX 1 & 4: Processing Speed (VR) & Energy (VE) ---
        # We rewrite the entire VR/VE blocks based on location type
        new_VR = {}
        new_VE = {}
        new_DE = {}
        
        for loc_id, loc_type in loc_types.items():
            # Set VR (Speed)
            new_VR[loc_id] = VR_SETTINGS.get(loc_type, 1.0e-7)
            
            # Set DE (Transmission Energy)
            new_DE[loc_id] = DE_SETTINGS.get(loc_type, 0.00001)
            
            # Keep VE proportional to speed (faster CPU = more power per cycle)
            if loc_type == 'iot':
                new_VE[loc_id] = 6.0e-7
            elif loc_type == 'edge':
                new_VE[loc_id] = 2.0e-7
            else: # cloud
                new_VE[loc_id] = 1.0e-7

        case['env']['VR'] = new_VR
        case['env']['VE'] = new_VE
        case['env']['DE'] = new_DE

        # --- FIX 2: Network Latency (DR) ---
        # We must iterate all possible links and assign DR based on the endpoints
        new_DR = {}
        all_ids = list(loc_types.keys())
        
        for u in all_ids:
            for v in all_ids:
                key = f"{u},{v}"
                type_u = loc_types[u]
                type_v = loc_types[v]
                
                if u == v:
                    new_DR[key] = 0.0
                    continue
                
                # Logic: Is this an IoT-Cloud link?
                is_iot_cloud = (type_u == 'iot' and type_v == 'cloud') or \
                               (type_u == 'cloud' and type_v == 'iot')
                
                if is_iot_cloud:
                    new_DR[key] = DR_SETTINGS['iot_to_cloud']
                else:
                    new_DR[key] = DR_SETTINGS['iot_to_edge'] # Default to fast

        case['env']['DR'] = new_DR

    # Save the result
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Done! Fixed dataset saved to '{output_file}'.")
    print("Please rename it to 'dataset.json' to use it.")

if __name__ == "__main__":
    fix_dataset()