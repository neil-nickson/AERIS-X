import os
import glob
import argparse
from aeris.config import ConfigManager
from aeris.analysis import ExperimentAnalyzer

def discover_phase8_datasets():
    print("PHASE 9 DATA DISCOVERY")
    print("----------------------")
    
    files = glob.glob("experiments/phase8_*.csv")
    valid_files = []
    rejected_files = []
    
    for f in files:
        if "test" in f or "exp_123" in f:
            rejected_files.append((f, "Test fixture"))
            continue
            
        if "_trial_summary" in f or "_analysis" in f:
            rejected_files.append((f, "Derived output file"))
            continue
            
        if "synthetic" in f:
            rejected_files.append((f, "Synthetic/test fixture"))
            continue
            
        if not os.path.exists(f):
            rejected_files.append((f, "File not found"))
            continue
            
        if os.path.getsize(f) == 0:
            rejected_files.append((f, "Empty file"))
            continue
            
        # Parse check
        try:
            with open(f, 'r') as fp:
                import csv
                reader = csv.DictReader(fp)
                headers = reader.fieldnames
                if not headers:
                    rejected_files.append((f, "Missing headers"))
                    continue
                    
                req = ["experiment_id", "condition", "trial_number", "data_quality", "temperature", "light", "vibration_index"]
                missing = [c for c in req if c not in headers]
                if missing:
                    rejected_files.append((f, f"Missing required columns: {missing}"))
                    continue
                    
                rows = list(reader)
                if len(rows) < 1:
                    rejected_files.append((f, "No data rows"))
                    continue
                    
                exp_id = rows[0].get("experiment_id", "")
                if exp_id.startswith("test_") or exp_id == "test_exp_123":
                    rejected_files.append((f, f"Contains synthetic test fixture: {exp_id}"))
                    continue
                    
            valid_files.append(f)
        except Exception as e:
            rejected_files.append((f, f"Malformed CSV: {e}"))
            
    valid_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"DISCOVERED PHASE 8 FILES: {len(files)}")
    print(f"VALID REAL EXPERIMENT FILES: {len(valid_files)}")
    print(f"EXCLUDED FILES: {len(rejected_files)}")
    
    if rejected_files:
        print("\nEXCLUDED FILE DETAILS:")
        for rf, reason in rejected_files:
            print(f"  - {rf} ({reason})")
            
    if valid_files:
        print("\nINCLUDED EXPERIMENT FILES:")
        for vf in valid_files:
            print(f"  - {vf}")
    
    return valid_files, rejected_files

def main():
    print("==========================================")
    print("AERIS-X PHASE 9: EXPERIMENT ANALYZER")
    print("==========================================")
    
    parser = argparse.ArgumentParser(description="AERIS-X Phase 9 Analysis")
    parser.add_argument("--method", type=str, default="pearson", choices=["pearson", "spearman"], help="Correlation method")
    args = parser.parse_args()
    
    valid_files, rejected_files = discover_phase8_datasets()
        
    if not valid_files:
        print("NO PHASE 8 EXPERIMENTAL DATA FOUND")
        return
        
    print(f"\nLoading {len(valid_files)} datasets...")
    
    config = ConfigManager()
    analyzer = ExperimentAnalyzer(config)
    
    analyzer.discovered_files = len(valid_files) + len(rejected_files)
    analyzer.included_files = valid_files
    analyzer.excluded_files = rejected_files
    
    analyzer.load_csvs(valid_files)
    analyzer.aggregate_trials()
    analyzer.compare_to_baseline()
    analyzer.calculate_correlations(method=args.method)
    analyzer.detect_confounders()
    
    txt_path, csv_path, json_path = analyzer.generate_outputs()
    
    print("\nAnalysis Complete.")
    print("==========================================")
    print(f"Text Report: {txt_path}")
    print(f"JSON Report: {json_path}")
    print(f"CSV Summary: {csv_path}")

    # Print comprehensive terminal report
    with open(txt_path, 'r') as f:
        print("\n--- REPORT SUMMARY ---")
        print(f.read())

if __name__ == "__main__":
    main()
