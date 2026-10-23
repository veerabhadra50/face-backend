-- FaceAttendance — Supabase / PostgreSQL schema
-- Run this in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
--
-- The backend also auto-creates these tables on startup via SQLAlchemy, so
-- running this manually is optional. Use it if you want the tables ready
-- before first launch, or to manage them from the Supabase UI.

-- Employees: one row per registered person. The 512-dim face embedding is
-- stored as JSON (a list of floats). No raw image is ever stored.
CREATE TABLE IF NOT EXISTS employees (
    id          SERIAL PRIMARY KEY,
    employee_id VARCHAR(64) UNIQUE NOT NULL,
    name        VARCHAR(128) NOT NULL,
    email       VARCHAR(128),
    department  VARCHAR(128),
    embedding   JSONB NOT NULL,
    created_at  TIMESTAMP DEFAULT (now() AT TIME ZONE 'utc')
);
CREATE INDEX IF NOT EXISTS ix_employees_employee_id ON employees (employee_id);

-- Attendance: one row per recognized check-in / check-out event.
CREATE TABLE IF NOT EXISTS attendance (
    id          SERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees (id) ON DELETE CASCADE,
    timestamp   TIMESTAMP DEFAULT (now() AT TIME ZONE 'utc'),
    event_type  VARCHAR(16) DEFAULT 'check_in',
    confidence  DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_attendance_timestamp ON attendance (timestamp);
CREATE INDEX IF NOT EXISTS ix_attendance_employee_id ON attendance (employee_id);

-- Visitors: one row per unrecognized face seen at the door.
CREATE TABLE IF NOT EXISTS visitors (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMP DEFAULT (now() AT TIME ZONE 'utc'),
    best_match_score DOUBLE PRECISION,
    note            TEXT
);
CREATE INDEX IF NOT EXISTS ix_visitors_timestamp ON visitors (timestamp);

-- OPTIONAL: Row Level Security.
-- The backend connects with the Postgres role (via DATABASE_URL / service_role),
-- which bypasses RLS. Only enable the policies below if you also plan to query
-- these tables directly from the frontend using the ANON key.
--
-- ALTER TABLE employees  ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE attendance ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE visitors   ENABLE ROW LEVEL SECURITY;
--
-- CREATE POLICY "read_all_attendance" ON attendance FOR SELECT USING (true);
-- CREATE POLICY "read_all_visitors"   ON visitors   FOR SELECT USING (true);
-- Note: never expose the employees.embedding column publicly.
