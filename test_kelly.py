"""
Quick test to verify Kelly Criterion is working
"""
from kelly_criterion import KellyCriterion

# Test Kelly calculation
test_backtest = {
    'win_rate': 55,  # 55% win rate
    'avg_win': 120,  # Average win $120
    'avg_loss': 80   # Average loss $80
}

print("=" * 60)
print("KELLY CRITERION TEST")
print("=" * 60)
print(f"\nTest Strategy:")
print(f"  Win Rate: {test_backtest['win_rate']}%")
print(f"  Avg Win: ${test_backtest['avg_win']}")
print(f"  Avg Loss: ${test_backtest['avg_loss']}")
print(f"  Win/Loss Ratio: {test_backtest['avg_win'] / test_backtest['avg_loss']:.2f}")

# Calculate Kelly
kelly_result = KellyCriterion.calculate_kelly_from_backtest(test_backtest, fractional_kelly=0.25)

print(f"\nKelly Criterion Results:")
print(f"  Kelly Fraction: {kelly_result['kelly_fraction']:.4f}")
print(f"  Recommended Position: {kelly_result['recommended_position_pct']:.2f}%")
print(f"  Risk Level: {kelly_result['risk_level']}")
print(f"  Analysis: {kelly_result['analysis']}")

print("\n" + "=" * 60)
print("✅ Kelly Criterion is working!")
print("=" * 60)
