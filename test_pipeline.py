#!/usr/bin/env python
"""Quick test of the updated pipeline with anomaly-aware metrics."""

if __name__ == "__main__":
    try:
        from src.evaluation.demo_evaluation import run_demo
        
        print("=" * 80)
        print("TESTING ANOMALY-AWARE PIPELINE")
        print("=" * 80)
        
        results = run_demo()
        
        print("\nModel Comparison Results:")
        print(results.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("SUCCESS! Pipeline working with anomaly-aware metrics")
        print("=" * 80)
        
        # Check for anomalies in results
        if "n_anomalies" in results.columns:
            print(f"\nAnomaly Detection Summary:")
            for idx, row in results.iterrows():
                print(f"  {row['model']:20} - {row['n_anomalies']:3} anomalies (downweighted)")
        
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        exit(1)
