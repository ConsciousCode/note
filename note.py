#!/usr/bin/env python3

from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, ContextManager, Final, Iterable, Iterator, Optional, NamedTuple
import re
import os

CONFIG: Final = "~/.config/notelog.conf"

## Defaults if config file is missing ##

DB_FILE: Final = "~/.local/share/notelog.db"
DB_NOTE_TABLE: Final = "notes"
DB_EDIT_TABLE: Final = "edits"

MAY: Final = {
    "water", "pee", "poop", "masturbate",
    "shower", "teeth", "nails", "shave",
    "sleep", "nap", "wake",
    "test"
}
MUST: Final = {
    "begin", "fail", "complete",
    "note", "dream", "health",
    "food", "coffee"
}
LIMIT: Final = {
    "coffee": {"begin", "complete"}
}

## Regex ##

UNITS: Final = '(?:secs?|seconds?|mins?|minutes?|hrs?|hours?|[smh])'
RANGE_RE: Final = re.compile(r"([\da-f]+)-([\da-f]+)")
DELTA_RE: Final = re.compile(rf"([+-]?)\s*(\d+)\s*({UNITS})")
TIME_RE: Final = re.compile(
    r"([01]?\d|2[0-3]):([0-5]?\d)(?::([0-5]?\d))?\s*([ap]m)?"
)
# RELative TimeStamp
RELTS_RE: Final = re.compile(rf'''
    (?:((?:[+-]?\s*\d+\s*{UNITS})+)(?:\s+ago\b)?)?
    \s*(?:([<>]|before|after)?\s*\b(\S+))?
''', re.X)

SCHEMA: Final = '''
CREATE TABLE IF NOT EXISTS {notes} (
    id INTEGER PRIMARY KEY,
    created_at INTEGER,
    deleted_at INTEGER,
    tag TEXT NOT NULL,
    note TEXT
);
'''

## Classes ##

class CmdError(RuntimeError):
    pass

class Config(NamedTuple):
    db_fn: str
    db_notes: str
    db_edits: str
    may: set[str]
    must: set[str]
    limit: dict[str, set[str]]

class NoteRow(NamedTuple):
    id: int
    created_at: int
    deleted_at: int
    tag: str
    note: str

class NoteData:
    '''
    A context manager for interacting with the note database.
    '''
    
    def __init__(self, config: Config):
        self.config = config
    
    def __enter__(self):
        import sqlite3
        self.db = sqlite3.connect(self.config.db_fn).__enter__()
        self.db.executescript(SCHEMA.format(
            notes=self.config.db_notes
        ))
        self.db.commit()
        return self
    
    def __exit__(self, *exc):
        self.db.__exit__(*exc)
        del self.db
    
    def tag_clauses(self, tag: Optional[str]):
        if tag is None: return [], ()
        if tag == "": return ["deleted_at IS NULL"], ()
        
        tags = set(map(str.strip, tag.split(",")))
        
        # Wildcard include-delete
        if "?" in tags: return [], ()
        
        conds = []
        params = []
        
        for tag in tags:
            if tag == "!":
                conds.append("deleted_at IS NOT NULL")
            elif tag.endswith("!"):
                conds.append("tag = ? AND deleted_at IS NOT NULL")
                params.append(tag[:-1])
            elif tag.endswith("?"):
                conds.append("tag = ?")
                params.append(tag[:-1])
            else:
                conds.append("tag = ? AND deleted_at IS NULL")
                params.append(tag)
        
        return conds, tuple(params)
    
    def query_note(self, query: str, *args):
        cur = self.db.execute(query, args)
        cur.row_factory = lambda c, r: NoteRow(*r)
        return cur
    
    def exec_commit(self, query: str, *args):
        cur = self.db.execute(query, args)
        self.db.commit()
        return cur
    
    def count(self, tag: Optional[str]):
        conds, params = self.tag_clauses(tag)
        where = f"WHERE {' OR '.join(conds)}" if conds else ""
        cur = self.db.execute(f"SELECT COUNT(*) FROM notes {where}", params)
        return cur.fetchone()[0]
    
    def most_recent(self, tag: str, limit: Optional[int]=1, note: Optional[str]=None) -> list[NoteRow]:
        conds, params = self.tag_clauses(tag)
        where = f"WHERE ({' OR '.join(conds)})" if conds else ""
        if note is not None:
            where += " AND note LIKE ?"
            params = (*params, note)
        lclause = limit and f"LIMIT {limit}" or ""
        return self.query_note(
            f"SELECT * FROM notes {where} ORDER BY created_at DESC {lclause}",
            *params
        ).fetchall()
    
    def by_id(self, id: Optional[int]) -> Optional[NoteRow]:
        return self.query_note(
            'SELECT * FROM notes WHERE id = ?', id
        ).fetchone()
    
    def by_range(self, start: int, end: int) -> list[NoteRow]:
        return self.query_note(
            "SELECT * FROM notes WHERE id BETWEEN ? AND ?",
            start, end
        ).fetchall()
    
    @staticmethod
    def around(op: Callable[[int, int], int]):
        def around_fn(self, tag: str):
            match self.most_recent(tag, 2):
                case []: return None
                case [a]: return op(a.created_at, 1)
                case [a, b, *_]:
                    return op(a.created_at, (a.created_at - b.created_at)//2)
                case _: raise NotImplementedError
        return around_fn
        
    before = around(lambda a, b: a - b)
    after = around(lambda a, b: a + b)
    
    def insert(self, when: int, tag: str, note: Optional[str]) -> Optional[int]:
        return self.exec_commit(
            "INSERT INTO notes (created_at, tag, note) VALUES (?, ?, ?)",
            when, tag, note
        ).lastrowid
    
    def edit(self, id: int, tag: str, note: Optional[str], ts: Optional[int]) -> Optional[NoteRow]:
        assign = ["deleted_at = NULL"]
        params = []
        if tag:
            assign.append("tag = ?")
            params.append(tag)
        
        if note is not None:
            if note == "":
                assign.append("note = NULL")
            else:
                assign.append("note = ?")
                params.append(note)
        
        if ts is not None:
            assign.append("created_at = ?")
            params.append(ts)
        
        print(assign)
        self.exec_commit(
            f"UPDATE notes SET {', '.join(assign)} WHERE id = ?",
            *params, id
        )
        return self.by_id(id)
    
    # Design note: No way to remove more than one note at a time or by tag.
    #  This is to prevent accidental deletion of a large number of notes. If
    #  I *really* need to do that, I'll open the sqlite shell.
    
    # I used to have a pop subcommand too, but once edit was added it became
    #  redundant. The whole point was to pop the last note if I mistyped it.
    
    def delete(self, id: int):
        return self.exec_commit(
            "UPDATE notes SET deleted_at = ? WHERE id = ?", inow(), id
        ).rowcount > 0
    
    def undelete(self, id: int):
        return self.exec_commit(
            "UPDATE notes SET deleted_at = NULL WHERE id = ?", id
        ).rowcount > 0

    def parse_offset(self, cmd: Optional[str]):
        if cmd is None:
            return None, 0
        
        ts = RELTS_RE.match(cmd.lower().strip())
        if ts is None:
            raise CmdError(f"Invalid time string: {cmd!r}")
        
        delta = sum(
            int(s + x)*{"s": 1, "m": 60, "h": 60*60}[y[0]]
                for s, x, y in DELTA_RE.findall(ts[1] or "")
        )
        
        match ts[2]:
            case "<"|"before":
                base = self.before(ts[3])
                delta = -delta
            case ">"|"after": base = self.after(ts[3])
            case None: delta = -delta
            case _: raise NotImplementedError
        
        if base := ts[3]:
            if dt := TIME_RE.match(base):
                hour = int(dt[1])
                base = datetime.now().replace(
                    minute=int(dt[2]),
                    second=int(dt[3] or 0)
                )
                ampm = dt[4]
                if ampm is None:
                    if base.hour > hour:
                        base = base.replace(base.day - 1)
                elif ampm == 'pm' and hour < 12:
                    hour += 12
                base = int(base.replace(hour=hour).timestamp())
            elif base in MUST or base in MAY:
                base = self.before(base)
            else:
                raise CmdError(f"Unknown tag {base!r}.")
        else:
            base = None
        
        return base, delta

## Utility functions ##

def plural(n, s):
    return s if n == 1 else s + "s"

def inow():
    return int(datetime.now().timestamp())

@contextmanager
def load_info(db_fn: Optional[str], config_fn: str):
    with NoteData(get_config(db_fn, config_fn)) as data:
        yield data

def unpack[*A](args: Iterable[str], *defaults: *A) -> tuple[str, ...]|tuple[*A]:
    return (*args, *defaults)[:len(defaults)] # type: ignore

def expected(name: str):
    raise CmdError(f"Expected a {name}.")

def hexid(s) -> int:
    try: return int(s, 16)
    except ValueError:
        raise CmdError(f"Invalid hex id {s!r}.")

def warn(msg):
    print(f"Warning: {msg}")

def check_overflow(rest):
    if rest: warn("Too many arguments.")

def get_config(db_fn: Optional[str], fn: str) -> Config:
    import tomllib
    try:
        with open(fn, 'rb') as f:
            data: dict[str, Any] = tomllib.load(f)
    except FileNotFoundError:
        data = {}
    
    db = data.get("database", {})
    note = data.get("note", {})
    
    return Config(
        os.path.expanduser(db.get("file", db_fn or DB_FILE)),
        db.get("note_table", DB_NOTE_TABLE),
        db.get("edit_table", DB_EDIT_TABLE),
        set(map(str.lower, note.get("may", MAY))),
        set(map(str.lower, note.get("must", MUST))),
        {
            k.lower(): set(map(str.lower, v))
                for k, v in note.get("limit", {}).items()
        }
    )

def time_components(dt: timedelta):
    if x := dt.days//30:
        yield f"{x} {plural(x, 'month')}"
    if x := dt.days%30:
        yield f"{x} {plural(x, 'day')}"
    d, r = divmod(dt.seconds, 3600)
    if x := d:
        yield f"{x} {plural(x, 'hour')}"
    if x := r // 60:
        yield f"{x} {plural(x, 'minute')}"
    if dt.seconds < 60:
        yield f"{dt.seconds} {plural(dt.seconds, 'second')}"
    yield "ago"

def print_row(row: Optional[NoteRow], ago=False):
    if row is None:
        return print("No note found.")
    if row.deleted_at:
        print('\x1b[2m', end='')
    dt = datetime.fromtimestamp(row.created_at)
    data = [f"{row.id:4x}", dt, row.tag]
    if row.note: data.append(row.note)
    print(*data, sep='\t')
    if ago:
        print(' ', *time_components(datetime.now() - dt))
    if row.deleted_at:
        print('\x1b[0m', end='')

## Subcommands ##

type Info = ContextManager[NoteData]

def subcmd_add(info: Info, *args: str):
    match args:
        case []: raise expected("tag")
        case [tag, *rest]: check_overflow(rest[2:])
        case _: raise NotImplementedError
    
    tag = tag.lower()
    note, dt = unpack(rest, None, None)
    
    if tag in MUST:
        if note is None:
            raise CmdError(f"Note {tag!r} requires a note.")
    elif tag not in MAY:
        raise CmdError(f"Unknown tag {tag!r}.")
    
    with info as data:
        if tag in data.config.limit:
            if note: note = note.lower()
            if note not in data.config.limit[tag]:
                raise CmdError(f"Tag {tag!r} note must be one of {data.config.limit[tag]}.")
        
        base, offset = data.parse_offset(dt)
        if base is None:
            base = int(datetime.now().timestamp())
        
        print_row(data.by_id(data.insert(base + offset, tag, note)))

def subcmd_show(info: Info, *args: str):
    match args:
        case []: raise expected("hex id")
        case [id, *rest]: check_overflow(rest)
        case _: raise NotImplementedError
    
    try:
        with info as data:
            for id in args[0].split(","):
                if m := RANGE_RE.match(id):
                    for row in data.by_range(hexid(m[1]), hexid(m[2])):
                        print_row(row)
                else:
                    print_row(data.by_id(int(id, 16)))
    except ValueError:
        raise CmdError("Invalid hex id.")

def subcmd_count(info: Info, tag: Optional[str]=None, *rest):
    check_overflow(rest)
    with info as data:
        print(data.count("" if tag == '-a' else tag))

def subcmd_last(info: Info,
        count: str|int|None=1,
        tag: str="",
        note: Optional[str]=None,
        *rest
    ):
    check_overflow(rest)
    if isinstance(count, str):
        if count.endswith("?"):
            count = count[:-1]
            tag += "?"
        
        if count == "-a":
            count = None
        else:
            try:
                count = int(count)
            except ValueError:
                if tag: note = tag
                count, tag = 1, count
            
            if count <= 0:
                raise CmdError("Count must be positive.")
    
    with info as data:
        if rows := data.most_recent(tag, count, note):
            for row in reversed(rows):
                print_row(row, True)
        else:
            print(f"No notes for {tag!r}.")

def subcmd_edit(info: Info, *args: str):
    match args:
        case []: raise expected("hex id")
        case [id, *rest]: check_overflow(rest[3:])
        case _: raise NotImplementedError
    
    tag, note, time = unpack(rest, "", None, "")
    try:
        ts = int(datetime.fromisoformat(time).timestamp()) if time else None
    except ValueError:
        raise CmdError(f"Invalid time {time!r}.")
    
    with info as data:
        print_row(data.edit(hexid(id), tag.lower(), note, ts))

def subcmd_delete(info: Info, *args: str):
    match args:
        case []: raise expected("hex id")
        case [id, *rest]: check_overflow(rest)
        case _: raise NotImplementedError
    
    with info as data:
        if data.delete(hexid(id)):
            print("Deleted note.")
        else:
            print_row(None)

def subcmd_undelete(info: Info, *args: str):
    match args:
        case []: raise expected("hex id")
        case [id, *rest]: check_overflow(rest)
        case _: raise NotImplementedError
    
    with info as data:
        if data.undelete(hexid(id)):
            print("Undeleted note.")
        else:
            print_row(None)

def usage(what: str=""):
    import inspect
    
    TAG_INFO = '''
            A special suffix ! can be used to query only deleted notes.
            Any tag can have a trailing ? to include deleted notes.
            Multiple tags can be separated by commas.'''
    TIME_INFO = '''
            Time can be specified in a number of ways:
            - [+-]N [sec/min/hour] ["ago"]  Time offset.
            - [<>]|before|after <tag>       Relative to the last tag.
            - HH:MM[:SS] [am/pm]            Explicit time.
            
            They are implicitly added together to form a final datetime.'''
    DOCS = {
        "": f'''
            usage: note [-h [cmd]] [-d DB] [-c CONFIG] cmd ...
            
            subcommands:
              add <tag> [note [dt]]  Add a note (implicit).
                <tag> [note [dt]]
              show <id>              Show a note by hex id.
              count <tag>            Count the tags noted.
              last <count> [tag]     Get last tagged notes.
              tags                   List all tags.
              edit <id> <note>       Edit a note by hex id.
              delete <id>            Delete a note by hex id.
              undelete <id>          Undelete a note by hex id.
              sql                    Open a sqlite3 shell.
              help [cmd]             Show this help message.
            {TAG_INFO}
            {TIME_INFO}
            
            options:
              -h, --help [cmd]     Show this help message and exit
              -d, --db DB          Database file
              -c, --config CONFIG  Config file
        ''',
        "add": f'''
            usage: note add <tag> [note [dt]]
                   note <tag> [note [dt]]
            
            Add a note to the database.
            {TAG_INFO}
            {TIME_INFO}
        ''',
        "show": '''
            usage: note show <id>
            
            Show a note by its hex id. Also allows id to be a comma-separated
            list of hex ids and ranges. Ex: note show 1,3,5-7
        ''',
        "count": f'''
            usage: note count [tag]
            
            Count the number of notes with the given tag.
            {TAG_INFO}
        ''',
        "tags": f'''
            usage: note tags
            
            Requires note: {', '.join(MUST)}
            Note optional: {', '.join(MAY)}
        ''',
        "last": f'''
            usage: note last <N> [tag]
                        
            Get the last N notes with the given tag. If N ends with a ?,
            implicitly include deleted notes.
            {TAG_INFO}
        ''',
        "edit": '''
            usage: note edit <id> [tag] [note] [time]
            
            Edit a note by its hex id. This does not check for tag validity,
            and will automatically undeleted the note if it was deleted.
        ''',
        "delete": '''
            usage: note delete <id>
            
            Delete a note by its hex id.
        ''',
        "sql": '''
            usage: note sql
            
            Open an sqlite3 shell on the database.
        ''',
        "help": '''
            usage: note help [cmd]
            
            Show the help message for a subcommand.
        '''
    }
    
    if doc := DOCS.get(what):
        return inspect.cleandoc(doc)
    doc = inspect.cleandoc(DOCS[""])
    return f"Unknown subcommand {what!r}\n\n{doc}."

def named_value(arg: str, it: Iterator[tuple[int, str]]):
    try: return next(it)
    except StopIteration:
        raise expected(f"value after {arg}")

def main(*argv: str):
    opts = {}
    it = iter(enumerate(argv, 1))
    try:
        while True:
            i, arg = next(it)
            match arg:
                case "-h"|"--help":
                    try:
                        i, h = next(it)
                    except StopIteration:
                        return print(usage())
                    
                    check_overflow(argv[i:])
                    return print(usage(h))
                
                case '-c'|'--config':
                    i, opts['config'] = named_value(arg, it)
                
                case "-d"|"--db":
                    i, opts['db'] = named_value(arg, it)
                
                case _:
                    break
        
        db_fn = opts.get("db")
        config_fn = opts.get("config", CONFIG)
        info = load_info(db_fn, config_fn)
        rest = [arg for _, arg in it]
        match arg:
            case "show": return subcmd_show(info, *rest)
            case "help":
                check_overflow(rest[1:])
                return print(usage(*rest[:1]))
            case "count": return subcmd_count(info, *rest)
            case "tags": return print(usage("tags"))
            case "last": return subcmd_last(info, *rest)
            case "edit": return subcmd_edit(info, *rest)
            case "delete": return subcmd_delete(info, *rest)
            case "sql":
                config = get_config(db_fn, config_fn)
                return os.execvp("sqlite3", ["sqlite3", config.db_fn])
            
            case "add": rest.pop(0)
            case _: rest.insert(0, arg)
        
        return subcmd_add(info, *rest)
    except BrokenPipeError:
        pass # head and tail close stdout
    except StopIteration:
        warn("Expected a command.")
        print(usage())
    except KeyboardInterrupt:
        print()
    except CmdError as e:
        warn(e)

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
