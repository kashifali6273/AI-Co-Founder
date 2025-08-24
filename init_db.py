import sqlite3

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Enable foreign key constraints
    c.execute("PRAGMA foreign_keys = ON")

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Saved ideas table
    c.execute('''
        CREATE TABLE IF NOT EXISTS saved_ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            startup_name TEXT NOT NULL,
            tagline TEXT,
            idea TEXT NOT NULL,
            tech_stack TEXT,
            sentiment TEXT,
            label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()


def update_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    # Add sentiment column if missing
    try:
        c.execute("ALTER TABLE saved_ideas ADD COLUMN sentiment TEXT")
        print("✅ Added sentiment column")
    except sqlite3.OperationalError:
        print("⚠️ Sentiment column already exists.")

    # Add label column if missing
    try:
        c.execute("ALTER TABLE saved_ideas ADD COLUMN label TEXT")
        print("✅ Added label column")
    except sqlite3.OperationalError:
        print("⚠️ Label column already exists.")

    # Add created_at column if missing
    try:
        c.execute("ALTER TABLE saved_ideas ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        print("✅ Added created_at column")
    except sqlite3.OperationalError:
        print("⚠️ created_at column already exists.")

    # Add updated_at column if missing
    try:
        c.execute("ALTER TABLE saved_ideas ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        print("✅ Added updated_at column")
    except sqlite3.OperationalError:
        print("⚠️ updated_at column already exists.")

    conn.commit()
    conn.close()
    print("✅ Database updated with sentiment, label, and timestamps in saved_ideas table")


if __name__ == '__main__':
    init_db()
    update_db()
