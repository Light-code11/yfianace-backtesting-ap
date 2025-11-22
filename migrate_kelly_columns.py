"""
Database migration to add Kelly Criterion columns
Run this to update existing database with Kelly columns
"""
import os
from sqlalchemy import create_engine, text, inspect
from database import DATABASE_URL, Base, BacktestResult

def migrate_database():
    """Add Kelly Criterion columns if they don't exist"""

    # Fix Railway PostgreSQL URL format (postgres:// -> postgresql://)
    db_url = DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    engine = create_engine(db_url)

    print("=" * 60)
    print("KELLY CRITERION DATABASE MIGRATION")
    print("=" * 60)
    print(f"Database: {db_url.split('@')[0]}...@{db_url.split('@')[1] if '@' in db_url else 'local'}")

    # Check if columns exist
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('backtest_results')]

    print(f"\nCurrent columns in backtest_results: {len(columns)}")

    kelly_columns = ['kelly_criterion', 'kelly_position_pct', 'kelly_risk_level']
    missing_columns = [col for col in kelly_columns if col not in columns]

    if not missing_columns:
        print("✅ All Kelly Criterion columns already exist!")
        print(f"   - kelly_criterion")
        print(f"   - kelly_position_pct")
        print(f"   - kelly_risk_level")
        print("\n" + "=" * 60)
        print("No migration needed!")
        print("=" * 60)
        return True

    print(f"\n⚠️  Missing columns: {missing_columns}")
    print("\n🔧 Adding Kelly Criterion columns...")

    with engine.connect() as conn:
        try:
            # Add missing columns
            if 'kelly_criterion' in missing_columns:
                conn.execute(text("ALTER TABLE backtest_results ADD COLUMN kelly_criterion FLOAT"))
                print("   ✅ Added kelly_criterion")

            if 'kelly_position_pct' in missing_columns:
                conn.execute(text("ALTER TABLE backtest_results ADD COLUMN kelly_position_pct FLOAT"))
                print("   ✅ Added kelly_position_pct")

            if 'kelly_risk_level' in missing_columns:
                conn.execute(text("ALTER TABLE backtest_results ADD COLUMN kelly_risk_level VARCHAR"))
                print("   ✅ Added kelly_risk_level")

            conn.commit()

            print("\n" + "=" * 60)
            print("✅ MIGRATION COMPLETE!")
            print("=" * 60)
            print("\n📝 Next Steps:")
            print("1. Run a NEW backtest to see Kelly Criterion in action")
            print("2. Old backtests will have NULL Kelly values (expected)")
            print("3. Check Streamlit Dashboard to see Kelly columns")

            return True

        except Exception as e:
            print(f"\n❌ Error during migration: {e}")
            conn.rollback()
            return False

if __name__ == "__main__":
    success = migrate_database()
    exit(0 if success else 1)
