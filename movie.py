#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional
import os
import re

import imdb
import psycopg2 as pg

SCHEMA = '''
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    imdb INTEGER UNIQUE,
    watched_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    rating INTEGER,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_movies_title
    ON movies USING GIN (to_tsvector('english', title));
CREATE TABLE IF NOT EXISTS todo (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    imdb INTEGER UNIQUE,
    created_at TIMESTAMP NOT NULL,
    notes TEXT
);
'''

DSN = 'postgresql:///movies'

@dataclass
class MovieRow:
    id: int
    title: str
    imdb: Optional[int]
    watched_at: Optional[datetime]
    created_at: datetime
    rating: Optional[int]
    notes: Optional[str]

    def print(self):
        print(
            f"{self.id:4x}", f"tt{self.imdb}", repr(self.title),
            "" if self.rating is None else self.rating,
            f"{self.watched_at:%Y-%m-%d}" if self.watched_at else "",
            sep='\t'
        )
        if self.notes:
            print(re.sub("^", "  ", self.notes))

@dataclass
class ToWatchRow:
    id: int
    title: str
    imdb: Optional[int]
    created_at: datetime
    notes: Optional[str]

    def print(self):
        print(f"{self.id:4x}", f"tt{self.imdb}", repr(self.title), sep='\t')
        if self.notes:
            print(re.sub("^", "  ", self.notes))

class Database:
    def __init__(self):
        pass
    
    def __enter__(self):
        self.conn = pg.connect(DSN)
        self.cur = self.conn.cursor()
        self.execute(SCHEMA)
        self.commit()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cur.close()
        self.conn.close()
    
    def execute(self, query: str, args: Optional[tuple]=None):
        self.cur.execute(query, args)

    def commit(self):
        self.conn.commit()

    def add_movie(self,
            title: str,
            tt: Optional[int],
            rating: Optional[int],
            created_at: datetime,
            watched_at: Optional[datetime],
            notes: Optional[str]
        ):
        self.execute('''
            SELECT * FROM movies WHERE imdb = %s
        ''', (tt,))
        if self.cur.fetchone():
            exit(f'Movie {title!r} already exists (tt{tt})')
        
        self.execute('''
            INSERT INTO movies (title, imdb, created_at, watched_at, rating, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
        ''', (title, tt, created_at, watched_at, rating, notes))
    
    def watch_movie(self, title: str, tt: Optional[int], notes: Optional[str]=None):
        now = datetime.now()
        self.execute('''
            INSERT INTO todo (title, imdb, created_at, notes)
                VALUES (%s, %s, %s, %s)
        ''', (title, tt, now, notes))
    
    def get_todo_list(self) -> list[ToWatchRow]:
        self.execute('''
            SELECT * FROM todo
        ''')
        return [ToWatchRow(*row) for row in self.cur.fetchall()]
    
    def search_movie(self, title: str) -> list[MovieRow]:
        self.execute('''
            SELECT * FROM movies WHERE title @@ %s
        ''', (title,))
        return [MovieRow(*row) for row in self.cur.fetchall()]
    
    def get_movies(self) -> list[MovieRow]:
        self.execute('''
            SELECT * FROM movies
        ''')
        return [MovieRow(*row) for row in self.cur.fetchall()]

def unpack[*A](args: Iterable[str], *defaults: *A) -> tuple[str, ...]|tuple[*A]:
    '''Unpack rest arguments with defaults and proper typing.'''
    return (*args, *defaults)[:len(defaults)] # type: ignore

def add_movie(argv: list[str]):
    if len(argv) < 1:
        return print('Usage: movie add TITLE [RATING] [WATCHED_AT] [NOTES]')
    
    title, *argv = argv
    rating, watched_at, notes = unpack(argv, None, None, None)
    
    if rating:
        try:
            rating = int(rating)
        except ValueError:
            exit('Rating must be an integer')
    else:
        rating = None
    
    created_at = datetime.now()

    if watched_at is None:
        watched_at = created_at
    elif watched_at:
        try:
            watched_at = datetime.fromisoformat(watched_at)
        except ValueError:
            exit('Invalid date format')
    else:
        watched_at = None
    
    im = imdb.IMDb()
    if not (results := im.search_movie(title)):
        exit('No results found')
    
    if (tt := results[0].getID()) is not None:
        tt = int(tt)
    
    with Database() as db:
        db.add_movie(title, tt, rating, created_at, watched_at, notes)
        db.commit()

def watch_movie(argv: list[str]):
    if len(argv) < 1:
        return print('Usage: movie add TITLE [RATING] [WATCHED_AT] [NOTES]')
    
    title, argv = argv[0], argv[1:]
    notes, = unpack(argv, None)

    im = imdb.IMDb()
    if not (results := im.search_movie(title)):
        exit('No results found')
    
    if (tt := results[0].getID()) is not None:
        tt = int(tt)
    
    with Database() as db:
        db.watch_movie(title, tt, notes)
        db.commit()

def todo_movie(argv: list[str]):
    with Database() as db:
        rows = db.get_todo_list()
    
    for row in rows:
        row.print()

def seen_movie(argv: list[str]):
    if len(argv) < 1:
        return print('Usage: movie seen QUERY')
    
    with Database() as db:
        for row in db.search_movie(argv[0]):
            row.print()

def dump_movies():
    with Database() as db:
        rows = db.get_movies()
    
    for row in rows:
        row.print()

def main(argv: list[str]):
    if not argv:
        return print('Usage: movie COMMAND [ARGS]')
    
    match argv[0]:
        case 'add': add_movie(argv[1:])
        case 'watch': watch_movie(argv[1:])
        case 'todo': todo_movie(argv[1:])
        case 'seen': seen_movie(argv[1:])
        case 'dump': dump_movies()
        case 'sql': os.execvp("psql", ["-d", DSN, *argv[1:]])
        case _:
            return print('Usage: movie COMMAND [ARGS]')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])