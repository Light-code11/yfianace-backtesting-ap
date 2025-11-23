"""Quick test of auto strategy generator with minimal settings"""
from auto_strategy_generator import AutoStrategyGenerator

print("Testing Auto Strategy Generator...")
print("This will take 2-3 minutes to backtest strategies\n")

generator = AutoStrategyGenerator()

# Test with very small batch and low requirements
results = generator.run_full_cycle(
    num_strategies=3,    # Only 3 strategies for quick test
    min_sharpe=0.5,      # Very low threshold (just testing)
    top_n=2              # Deploy top 2 if any pass
)

print("\n" + "=" * 70)
print("TEST RESULTS:")
for key, value in results.items():
    print(f"  {key}: {value}")
print("=" * 70)
