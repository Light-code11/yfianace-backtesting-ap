"""
Startup script for Railway deployment
Handles PORT environment variable properly
"""
import os
import uvicorn

if __name__ == "__main__":
    # Run database migration for Kelly Criterion columns
    print("Running database migration...")
    try:
        from migrate_kelly_columns import migrate_database
        migrate_database()
        print("✅ Migration complete!\n")
    except Exception as e:
        print(f"⚠️ Migration warning: {e}")
        print("Continuing with server startup...\n")

    # Get PORT from environment or default to 8000
    port = int(os.getenv("PORT", "8000"))

    print(f"Starting server on port {port}...")

    # Run the application
    uvicorn.run(
        "trading_platform_api:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
