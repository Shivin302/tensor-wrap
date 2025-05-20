import argparse
import time
import random
from tqdm import tqdm

def simulate_inference(model_name):
    print(f"\nRunning script to profile {model_name}")
    time.sleep(1)
    print(f"\nLoading {model_name}...")
    time.sleep(1)
    
    print("\nRunning inference...")
    latency = 73 if model_name == "optimized_LLM.pt" else 261
    tsteps = 100
    for _ in tqdm(range(tsteps), desc="Inference", unit="step"):
        time.sleep(latency / tsteps / 100)
    
    print(f"\nInference complete! Latency of the model is {latency} miliseconds!")
    print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate running a  LLM")
    parser.add_argument("model", choices=["LLM.pt", "optimized_LLM.pt"], help="Model to run")
    args = parser.parse_args()
    
    simulate_inference(args.model)