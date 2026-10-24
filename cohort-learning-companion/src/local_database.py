"""Start/stop the project-local PostgreSQL runtime installed under .local-db."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
from urllib.parse import quote, urlsplit

from dotenv import dotenv_values, set_key
import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local-db'
BIN = LOCAL / 'runtime' / 'Library' / 'bin'
DATA = LOCAL / 'data'


def run(*args, check=True):
    environment = os.environ.copy()
    environment['PATH'] = str(BIN) + os.pathsep + environment.get('PATH', '')
    # pg_ctl's server child can inherit pipe handles on Windows; using
    # DEVNULL for startup prevents communicate() waiting on a live server.
    background = Path(args[0]).name == 'pg_ctl.exe' and 'start' in args
    output = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL} if background else {'capture_output': True}
    result = subprocess.run([str(args[0]), *map(str, args[1:])], env=environment,
                            **output, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if check and result.returncode:
        raise RuntimeError((result.stderr or '').strip() or (result.stdout or '').strip() or 'Database command failed')
    return result


def start():
    if not (BIN / 'pg_ctl.exe').exists():
        raise RuntimeError('Project-local PostgreSQL runtime is missing; see RETRIEVAL.md')
    LOCAL.mkdir(exist_ok=True)
    config = dotenv_values(ROOT / '.env')
    initialized = (DATA / 'PG_VERSION').exists()
    if initialized:
        url = config.get('DATABASE_URL', '')
        target = urlsplit(url)
        if target.hostname != '127.0.0.1' or not target.username or not target.path.strip('/'):
            raise RuntimeError('Existing local cluster requires its saved DATABASE_URL in .env; credentials were not changed')
        port, database = target.port or 5432, target.path.strip('/')
    else:
        # Do not overwrite an existing remote database configuration.
        old_url = config.get('DATABASE_URL') or ''
        if old_url and urlsplit(old_url).hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise RuntimeError('DATABASE_URL points to another host; local setup will not overwrite it')
        port = 5432
        database = 'cohort_db'
        user, password = 'cohort_admin', secrets.token_urlsafe(32)
        password_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=LOCAL, delete=False) as stream:
                password_file = Path(stream.name)
                stream.write(password + '\n')
            run(BIN / 'initdb.exe', '-D', DATA, '-U', user, '--encoding=UTF8', '--locale=C',
                '--auth-host=scram-sha-256', '--auth-local=scram-sha-256', '--pwfile', password_file)
        finally:
            if password_file:
                password_file.unlink(missing_ok=True)
        url = f'postgresql://{user}:{quote(password, safe="")}@127.0.0.1:{port}/{database}'
        set_key(str(ROOT / '.env'), 'DATABASE_URL', url)
    if run(BIN / 'pg_ctl.exe', '-D', DATA, 'status', check=False).returncode != 0:
        run(BIN / 'pg_ctl.exe', '-D', DATA, '-l', LOCAL / 'postgres.log',
            '-o', f'-h 127.0.0.1 -p {port}', '-w', '-t', '30', 'start')
    admin = urlsplit(url)
    admin_url = admin._replace(path='/postgres').geturl()
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as conn:
        if not conn.execute('SELECT 1 FROM pg_database WHERE datname=%s', (database,)).fetchone():
            conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(database)))
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
    print(f'PostgreSQL ready at 127.0.0.1:{port}/{database}; pgvector {version}. DATABASE_URL saved in .env.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'stop'], default='start', nargs='?')
    args = parser.parse_args()
    try:
        if args.command == 'start':
            start()
        elif (DATA / 'PG_VERSION').exists():
            if run(BIN / 'pg_ctl.exe', '-D', DATA, 'status', check=False).returncode == 0:
                run(BIN / 'pg_ctl.exe', '-D', DATA, '-m', 'fast', '-w', 'stop')
            print('Project-local PostgreSQL stopped.')
    except (RuntimeError, psycopg.Error, ValueError) as exc:
        # Do not print a URL or password from connection exceptions.
        if isinstance(exc, psycopg.Error):
            parser.exit(1, 'Local PostgreSQL setup failed; check .local-db/postgres.log and saved DATABASE_URL.\n')
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
