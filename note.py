#!/usr/bin/env python3

from datetime import datetime, timedelta
from typing import Any, Callable, Final, Iterable, Iterator, LiteralString, Optional, NamedTuple, cast
import re
import os
import sys

import psycopg as pg
from psycopg.rows import class_row

CONFIG: Final = "~/.config/notelog.toml"

## Defaults if config file is missing ##

DB_DSN: Final = "postgresql://notelog"
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

EDITOR: Final = os.environ.get("EDITOR", "nano")

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

## Database schema ##
SCHEMA: Final = '''
CREATE TABLE IF NOT EXISTS edits (
    edit_id INTEGER PRIMARY KEY,
    last_edit INTEGER,
    noted_at TIMESTAMP,
    modified_at TIMESTAMP NOT NULL, /* edit timestamp */
    tag TEXT NOT NULL,
    note TEXT,
    
    FOREIGN KEY(last_edit) REFERENCES edits(edit_id)
);
CREATE TABLE IF NOT EXISTS notes (
    note_id INTEGER PRIMARY KEY,
    last_edit INTEGER,
    noted_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP, /* first created */
    deleted_at TIMESTAMP,
    tag TEXT NOT NULL,
    note TEXT,
    
    FOREIGN KEY(last_edit) REFERENCES edits(edit_id)
);
'''

## Utility functions ##

def inow():
    return int(datetime.now().timestamp())

def unpack[*A](args: Iterable[str], *defaults: *A) -> tuple[str, ...]|tuple[*A]:
    '''Unpack rest arguments with defaults and proper typing.'''
    return (*args, *defaults)[:len(defaults)] # type: ignore

def expected(name: str):
    raise CmdError(f"Expected a {name}.")

def hexid(s) -> int:
    try: return int(s, 16)
    except ValueError:
        raise CmdError(f"Invalid hex id {s!r}.")

def warn(msg):
    print(f"Warning: {msg}", file=sys.stderr)

def check_overflow(rest):
    if rest: warn("Too many arguments.")

def time_components(dt: timedelta):
    '''Convert a timedelta to human-readable time components.'''
    def plural(n, s):
        b = f"{n} {s}"
        return b if n == 1 else f"{b}s"
    
    if x := dt.days//30:
        yield plural(x, 'month')
    if x := dt.days%30:
        yield plural(x, 'day')
    d, r = divmod(dt.seconds, 3600)
    if x := d:
        yield plural(x, 'hour')
    if x := r // 60:
        yield plural(x, 'minute')
    if dt.seconds < 60: 
        yield plural(dt.seconds, 'second')
    yield "ago"

def bash_quote(text: Optional[str]):
    if not text: return '""'
    if not re.search(r'''[$!\\'"`\s]''',text):
        return text
    if not re.search(r'[$!\\"`]', text):
        return f'"{text}"'
    return f"'{text.replace("'", "'\\''")}'"

## Classes ##

class CmdError(RuntimeError):
    '''Error for wrapping expected command errors, eg formatting.'''

class Config(NamedTuple):
    source: str
    editor: str
    dsn: str # Database source name
    may: set[str]
    must: set[str]
    default: dict[str, str]
    limit: dict[str, set[str]]

class EditRow(NamedTuple):
    edit_id: int
    last_edit: Optional[int]
    noted_at: Optional[int]
    modified_at: int
    tag: str
    note: Optional[str]
    
    def print(self):
        print(
            f"  \33[2m{self.edit_id:4x}", self.tag,
            bash_quote(self.note),
            datetime.fromtimestamp(self.modified_at).isoformat(),
            sep='\t',
            end='\33[m\n'
        )

class NoteRow(NamedTuple):
    note_id: int
    last_edit: Optional[int]
    noted_at: datetime
    created_at: Optional[int]
    deleted_at: Optional[int]
    tag: str
    note: Optional[str]
    
    def print(self, ago=False):
        if self.deleted_at:
            print('\033[2m', end='')
        dt = self.noted_at
        data = [f"{self.note_id:4x}", self.tag, bash_quote(self.note), dt.isoformat()]
        if self.last_edit: data.append("(edited)")
        print(*data, sep='\t')
        if ago:
            print(' ', *time_components(datetime.now() - dt))
        if self.deleted_at:
            print('\033[m', end='')

class NoteData:
    '''
    A context manager for interacting with the note database.
    '''
    
    def __init__(self, config: Config):
        self.config = config
    
    def __enter__(self):
        self.db = pg.connect(self.config.dsn).__enter__()
        self.db.cursor().execute(SCHEMA)
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
                conds.append("tag = %s AND deleted_at IS NOT NULL")
                params.append(tag[:-1])
            elif tag.endswith("?"):
                conds.append("tag = %s")
                params.append(tag[:-1])
            else:
                conds.append("tag = %s AND deleted_at IS NULL")
                params.append(tag)
        
        return conds, tuple(params)
    
    def query[T](self, kind: type[T], query: LiteralString, *args):
        cur = self.db.cursor(row_factory=class_row(kind))
        cur.execute(query, args)
        return cur

    def query_note(self, query: LiteralString, *args):
        return self.query(NoteRow, query, *args)
    
    def query_edit(self, query: LiteralString, *args):
        return self.query(EditRow, query, *args)
    
    def execute(self, query: LiteralString, *args):
        cur = self.db.cursor()
        cur.execute(query, args)
        return cur
    
    def count(self, tag: Optional[str]):
        conds, params = self.tag_clauses(tag)
        where = f"WHERE {' OR '.join(conds)}" if conds else ""
        cur = self.db.execute(
            cast(LiteralString, f"SELECT COUNT(*) FROM notes {where}"), params
        )
        if (row := cur.fetchone()) is None:
            raise RuntimeError("Failed to count notes.")
        return row[0]
    
    def most_recent(self, tag: str, limit: Optional[int]=1, note: Optional[str]=None) -> list[NoteRow]:
        conds, params = self.tag_clauses(tag)
        where = f"WHERE ({' OR '.join(conds)})" if conds else ""
        if note is not None:
            where += " AND note LIKE %s"
            params = (*params, note)
        lclause = limit and f"LIMIT {limit}" or ""
        return self.query_note(
            cast(LiteralString,
                f"SELECT * FROM notes {where} ORDER BY noted_at DESC {lclause}"
            ),
            *params
        ).fetchall()
    
    def by_id(self, id: int) -> Optional[NoteRow]:
        return self.query_note(
            'SELECT * FROM notes WHERE note_id = %s', id
        ).fetchone()
    
    def by_range(self, start: int, end: int) -> list[NoteRow]:
        return self.query_note(
            "SELECT * FROM notes WHERE note_id BETWEEN %s AND %s",
            start, end
        ).fetchall()
    
    @staticmethod
    def around(op: Callable[[int, int], int]):
        def around_fn(self, tag: str, ensure=False):
            match self.most_recent(tag, 2):
                case []:
                    if ensure:
                        raise CmdError(f"No notes found for {tag!r}.")
                    return None
                case [a]: return op(a.created_at, 1)
                case [a, b, *_]:
                    return op(a.created_at, (a.created_at - b.created_at)//2)
                case _: raise NotImplementedError
        return around_fn
        
    before = around(lambda a, b: a - b)
    after = around(lambda a, b: a + b)
    
    def insert(self, when: datetime, tag: str, note: Optional[str]) -> int:
        cur = self.execute('''
            INSERT INTO notes (noted_at, created_at, tag, note)
                VALUES (%s, %s, %s, %s)
                RETURNING note_id
            ''', when, datetime.now(), tag, note
        ).fetchone()
        if cur is None:
            raise RuntimeError("Failed to insert note.")
        return cur[0]
    
    def edit(self, id: int, tag: str, note: Optional[str], ts: Optional[datetime]) -> Optional[NoteRow]:
        # Edits automatically undelete the note.
        assign = ["deleted_at = NULL"]
        params = []
        if tag:
            assign.append("tag = %s")
            params.append(tag)
        
        if note is not None:
            if note == "":
                assign.append("note = NULL")
            else:
                assign.append("note = %s")
                params.append(note)
        
        if ts is not None:
            assign.append("noted_at = %s")
            params.append(ts)
        
        self.db.execute("BEGIN")
        self.db.execute('''
            INSERT INTO edits
                (last_edit, noted_at, modified_at, tag, note)
                SELECT last_edit, noted_at, %s AS modified_at, tag, note
                    FROM notes WHERE note_id = %s
        ''', (inow(), id))
        self.db.execute(cast(LiteralString, f'''
            UPDATE notes SET
                last_edit = last_insert_rowid(),
                {', '.join(assign)}
                WHERE note_id = %s
        '''), (*params, id))
        self.db.commit()
        return self.by_id(id)
    
    def get_edit(self, id: int) -> Optional[EditRow]:
        return self.query_edit(
            "SELECT * FROM edits WHERE edit_id = %s", id
        ).fetchone()
    
    def edits_of(self, id: int) -> Iterable[EditRow]:
        cur = self.by_id(id)
        if cur is None:
            return
        
        while last_edit := cur.last_edit:
            cur = self.query_edit(
                "SELECT * FROM edits WHERE edit_id = %s",
                last_edit
            ).fetchone()
            if cur is None:
                raise RuntimeError("Edit chain broken.")
            yield cur
    
    # Design note: No way to remove more than one note at a time or by tag.
    #  This is to prevent accidental deletion of a large number of notes. If
    #  I *really* need to do that, I'll open the sqlite shell.
    
    # I used to have a pop subcommand too, but once edit was added it became
    #  redundant. The whole point was to pop the last note if I mistyped it.
    
    def delete(self, id: int):
        return self.query_note("""
            UPDATE notes SET deleted_at = %s WHERE note_id = %s
            RETURNING *
        """, datetime.now(), id).fetchone()
    
    def parse_offset(self, cmd: Optional[str]) -> tuple[Optional[datetime], timedelta]:
        if cmd is None:
            return None, timedelta(0)
        
        try:
            return datetime.fromisoformat(cmd), timedelta(0)
        except ValueError:
            pass
        
        # dt [<> tag]
        if not (ts := RELTS_RE.match(cmd.lower().strip())):
            raise CmdError(f"Invalid time string: {cmd!r}")
        
        delta = sum(( # Add up time deltas
            int(s + x)*{
                "s": timedelta(seconds=1),
                "m": timedelta(minutes=1),
                "h": timedelta(hours=1)
            }[y[0]] for s, x, y in DELTA_RE.findall(ts[1] or "")),
            timedelta()
        )
        
        if sb := ts[3]:
            if dt := TIME_RE.match(sb):
                hour = int(dt[1])
                base = datetime.now().replace(minute=int(dt[2]))
                if sec := dt[3]:
                    base = base.replace(second=int(sec))
                # Adjust for 12-hour time
                match dt[4]:
                    case 'am':
                        if 12 <= base.hour >= hour + 12:
                            hour += 12
                    case 'pm':
                        if hour < 12:
                            hour += 12
                    case _:
                        now_hour = base.hour
                        if now_hour >= 12:
                            if hour < now_hour - 12:
                                hour += 12
                        elif hour < now_hour:
                            hour += 12
                base = int(base.replace(hour=hour).timestamp())
            elif sb in self.config.may or sb in self.config.must:
                match ts[2]:
                    case ">"|"after": base = self.after(ts[3], ensure=True)
                    case "<"|"before"|_:
                        base = self.before(ts[3], ensure=True)
                        delta = -delta
            else:
                raise CmdError(f"Unknown time {sb!r}.") from None
        else:
            base = None
        
        return base, delta

class NoteApp:
    def __init__(self, *rest, help=None, config=None, dsn=None, force=False):
        self.rest = rest
        self.help = help
        self.config = config or CONFIG
        self.dsn = dsn or DB_DSN
        self.force = force
    
    @classmethod
    def argparse(cls, *argv: str):
        '''Build the app from command line arguments.'''
        
        def named_value(arg: str, it: Iterator[tuple[int, str]]):
            try: return next(it)
            except StopIteration:
                raise expected(f"value after {arg}")
        
        try:
            opts = {}
            it = iter(enumerate(argv, 1))
            while True:
                i, arg = next(it)
                match arg:
                    case "-h"|"--help":
                        try:
                            i, h = next(it)
                        except StopIteration:
                            opts['help'] = ''
                            break
                        
                        check_overflow(argv[i:])
                        opts['help'] = h
                        break
                    
                    case '-c'|'--config':
                        i, opts['config'] = named_value(arg, it)
                    
                    case "-d"|"--db":
                        i, opts['db'] = named_value(arg, it)
                    
                    case "-f"|"--force":
                        opts['force'] = True
                    
                    case _:
                        break
            
            return cls(arg, *(arg for _, arg in it), **opts)
        except StopIteration:
            return None
    
    def get_config(self):
        import tomllib
        try:
            with open(os.path.expanduser(self.config), 'r') as f:
                source = f.read()
                data: dict[str, Any] = tomllib.loads(source)
        except FileNotFoundError:
            import inspect
            import json
            source = inspect.cleandoc(f'''
                ## Generated from defaults ##
                editor = {json.dumps(EDITOR)}
                
                [database]
                dsn = {json.dumps(self.dsn)}
                
                [note]
                # May have a note
                may = {json.dumps(list(MAY))}
                
                # Must have a note
                must = {json.dumps(list(MUST))}
                
                [note.default]
                
                # Must have one of these notes
                [note.limit]
                coffee = {json.dumps(list(LIMIT['coffee']))}
            ''')
            data = {}
        
        db = data.get("database", {})
        note = data.get("note", {})
        
        return Config(
            source,
            data.get("editor", EDITOR),
            os.path.expanduser(db.get("dsn", self.dsn)),
            set(map(str.lower, note.get("may", MAY))),
            set(map(str.lower, note.get("must", MUST))),
            {k.lower(): v for k, v in note.get("default", {}).items()},
            {
                k.lower(): set(map(str.lower, v))
                    for k, v in note.get("limit", {}).items()
            }
        )
    
    def info(self):
        return NoteData(self.get_config())
    
    def tag_info(self):
        # Don't enter so we don't connect to the db
        data = self.info()
        return data.config.must, data.config.may
    
    def usage(self, what: str=""):
        '''
        usage: {name} [-h [cmd]] [-d DB] [-c CONFIG] [-f] cmd ...
        
        subcommands:
        add <tag> [note [dt]]  Add a note (implicit).
            <tag> [note [dt]]
        config ["edit"]        Show or edit the configuration file.
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
            -h, --help [cmd]     Show this help message and exit.
            -d, --db DB          Database file.
            -c, --config CONFIG  Config file.
            -f, --force          Ignore note requirements.
        '''
        
        import inspect
        
        must, may = self.tag_info()
        if what == "tags":
            return f'''
                Requires note: {', '.join(must)}
                Optional note: {', '.join(may)}
            '''
        
        if what == "":
            doc = inspect.cleandoc(self.usage.__doc__ or "")
        elif sub := getattr(self, f"subcmd_{what}", None):
            doc = inspect.cleandoc(sub.__doc__ or "")
            doc = "usage: {name} " + f"{what} {doc}"
        else:
            doc = inspect.cleandoc(self.usage.__doc__ or "")
            doc = f"Unknown subcommand {what!r}\n\n{doc}"
        
        return doc.format(
            name=os.path.basename(sys.argv[0]),
            TAG_INFO=inspect.cleandoc('''
                A special suffix ! can be used to query only deleted notes. Any tag can have a trailing ? to include deleted notes. Multiple tags can be separated by commas.
            '''),
            TIME_INFO=inspect.cleandoc('''
                Time can be specified in a number of ways:
                - [+-]N [sec/min/hour] ["ago"]  Time offset.
                - [<>]|before|after <tag>       Relative to the last tag.
                - HH:MM[:SS] [am/pm]            Explicit time.
                
                They are implicitly added together to form a final datetime.
            ''')
        )
    
    def run(self):
        if self.help is not None:
            check_overflow(self.rest)
            return print(self.usage(self.help))
        
        if not self.rest:
            raise expected("subcommand")
        
        if subcmd := getattr(self, f"subcmd_{self.rest[0]}", None):
            return subcmd(*self.rest[1:])
        else:
            return self.subcmd_add(*self.rest)
    
    def subcmd_add(self, *args: str):
        '''
        <tag> [note [dt]]
            note <tag> [note [dt]]
        
        Add a note to the database.
        
        {TAG_INFO}
        
        {TIME_INFO}
        '''
        match args:
            case []: raise expected("tag")
            case [tag, *rest]: check_overflow(rest[2:])
            case _: raise NotImplementedError
        
        tag = tag.lower()
        note, dt = unpack(rest, None, None)
        with self.info() as data:
            if tag in data.config.must:
                if note is None and not self.force:
                    raise CmdError(f"Note {tag!r} requires a note.")
            elif tag in data.config.may:
                if note is None:
                    if (n := data.config.default.get(tag)) is not None:
                        note = str(n)
            elif not self.force:
                raise CmdError(f"Unknown tag {tag!r}.")
            
            if tag in data.config.limit:
                if note: note = note.lower()
                if note not in data.config.limit[tag] and not self.force:
                    raise CmdError(f"Tag {tag!r} note must be one of {data.config.limit[tag]}.")
            elif dt is None:
                # If no time is given, check if the note is a relative time.
                #  This only applies to may tags which can have note=None
                if tag in data.config.may:
                    if m := RELTS_RE.match(note or ""):
                        if m[3] is None:
                            note, dt = None, note
            
            base, offset = data.parse_offset(dt)
            if base is None:
                base = datetime.now()
            
            if note := data.by_id(data.insert(base + offset, tag, note)):
                note.print()
            else:
                warn("Failed to add note.")
    
    def subcmd_config(self, what: str="", *_):
        '''
        ["edit"]
        
        Show the configuration file or edit it. Respects EDITOR env.
        '''
        
        with self.info() as data:
            if what == "edit":
                editor = data.config.editor
                return os.execvp(editor, [
                    editor, os.path.expanduser(self.config)
                ])
            else:
                print(data.config.source)
    
    def subcmd_show(self, *args: str):
        '''
        <id>
        
        Show a note by its hex id. Also allows id to be a comma-separated list of hex ids and ranges. Ex: note show 1,3,5-7
        '''
        match args:
            case []: raise expected("hex id")
            case [id, *rest]: check_overflow(rest)
            case _: raise NotImplementedError
        
        with self.info() as data:
            for id in args[0].split(","):
                if m := RANGE_RE.match(id):
                    rowit = data.by_range(hexid(m[1]), hexid(m[2]))
                elif row := data.by_id(hexid(id)):
                    rowit = [row]
                else:
                    break
                
                for row in rowit:
                    row.print()
                    for edit in data.edits_of(row.note_id):
                        edit.print()

    def subcmd_count(self, tag: Optional[str]=None, *rest):
        '''
        [tag]
        
        Count the number of notes with the given tag.
        
        {TAG_INFO}
        '''
        check_overflow(rest)
        with self.info() as data:
            print(data.count("" if tag == '-a' else tag))
    
    def subcmd_tags(self, *rest: str):
        return print(self.usage("tags"))

    def subcmd_last(self,
            count: str|int|None=1,
            tag: str="",
            note: Optional[str]=None,
            *rest
        ):
        '''
        [count] [tag [note]]
                    
        Get the last N notes with the given tag. If N ends with a ?, implicitly include deleted notes.
        {TAG_INFO}
        '''
        check_overflow(rest)
        if isinstance(count, str):
            if m := re.match(r"(?:(\d+)|([^?!]+))([?!]*)", count):
                num, st, ex = m.groups()
                if num:
                    count = int(num)
                    if count <= 0:
                        raise CmdError("Count must be positive.")
                else:
                    if st == '-a':
                        count = None
                    else:
                        count = 1
                        tag, note = st, tag or None
                
                if "!" in ex: tag += "!"
                elif "?" in ex: tag += "?"
            else:
                count = None
        
        with self.info() as data:
            if rows := data.most_recent(tag, count, note):
                for row in reversed(rows):
                    row.print(True)
            else:
                print("No notes found.")

    def subcmd_edit(self, *args: str):
        '''
        <id> [tag] [note] [time]
        
        Edit a note by its hex id. This does not check for tag validity, and will automatically undeleted the note if it was deleted.
        '''
        match args:
            case []: raise expected("hex id")
            case [id, *rest]: check_overflow(rest[3:])
            case _: raise NotImplementedError
        
        tag, note, time = unpack(rest, "", None, "")
        try:
            ts = datetime.fromisoformat(time) if time else None
        except ValueError:
            raise CmdError(f"Invalid time {time!r}.")
        
        with self.info() as data:
            if note := data.edit(hexid(id), tag.lower(), note, ts):
                note.print()
            else:
                warn("No notes found.")

    def subcmd_delete(self, *args: str):
        '''
        <id>
        
        Delete a note by its hex id.
        '''
        match args:
            case []: raise expected("hex id")
            case [id, *rest]: check_overflow(rest)
            case _: raise NotImplementedError
        
        with self.info() as data:
            if note := data.delete(hexid(id)):
                print("Deleted note.")
                note.print()
            else:
                warn("No notes found.")
    
    def subcmd_sql(self, *rest: str):
        '''
         
        
        Open an sqlite3 shell on the database.
        '''
        config = self.get_config()
        return os.execvp("sqlite3", ["sqlite3", config.dsn, *rest])
    
    def subcmd_help(self, *rest: str):
        '''
        [cmd]
        
        Show the help message for a subcommand.
        '''
        check_overflow(rest[1:])
        return print(self.usage(*rest[:1]))

def main(*argv: str):
    try:
        if app := NoteApp.argparse(*argv):
            app.run()
        else:
            warn("Expected a command.")
            print(NoteApp().usage())
    except BrokenPipeError:
        pass # head and tail close stdout
    except KeyboardInterrupt:
        print()
    except CmdError as e:
        print("Error:", e, file=sys.stderr)

if __name__ == "__main__":
    main(*sys.argv[1:])
