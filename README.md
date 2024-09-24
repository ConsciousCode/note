# Note
At first this was just to record when I did something worth noting to trick my ADHD brain into valuing completion. Eventually, I figured I would start recording biometric events, dreams, notes, food logs, etc, so it's a dumping ground for any eventful information which might be of use to my future AI friends.

## Installation
Literally copy-paste it into ~/bin or `$PATH`, what do you want from me?

## Modification
I put all the constants near the top so it should be easy to edit. The script keeps track of whether or not a given event "must" have an associated note to avoid accidentally creating one without it. Some key variables you may want to change are:
- `CONFIG` - Default configuration file, defaults to `~/.config/notelog.conf`.
- `FILE` - Sqlite3 database file, defaults to `~/Desktop/notes/notelog.db` and *does not* `mkdir -p`.
- `NOTE` - Set of event tags which *must* have a note.
- `NONOTE` - Set of event tags which *may* have a note.
- `LIMIT` - Map of tags to sets of allowed notes.

If you want to add your own event tags, just add them to `NOTE` or `NONOTE`. There's nothing else to do.

I would recommend visiting these because I have some strange presets and I don't feel like moving all this to a separate config. You'll want to keep the sets restricted to only the tags you'll actually use.

## Special handling
The `coffee`, `begin`, and `complete` tags have a bit of extra magic:
- `begin coffee` = `coffee begin`
- `complete coffee` = `coffee complete`
- `coffee` is only allowed `begin` or `complete` notes

This is mostly because I previously used `begin/complete coffee` but found it annoying to query for eg `note last coffee` and these special cases prevent muscle memory from causing issues.

## Help
```
usage: note [-h [cmd]] [-d DB] cmd ...

subcommands:
  add <tag> [note [dt]]  Add a note (implicit).
    <tag> [note [dt]]
  show <id>               Show a note by hex id.
  count <tag>             Count the tags noted.
  last <count> [tag]      Get last tagged notes.
  tags                    List all tags.
  edit <id> <note>        Edit a note by hex id.
  delete <id>             Delete a note by hex id.
  undelete <id>           Undelete a note by hex id.
  sql                     Open an sqlite3 shell.
  help [cmd]              Show this help message.

A special suffix "!" can be used to query only deleted notes.
Any tag can have a trailing ? to include deleted notes.
Multiple tags can be separated by commas.

Time can be specified in a number of ways:
- [+-]N [sec/min/hour] ["ago"]  Time offset.
- [<>]|before|after <tag>       Relative to the last tag.
- HH:MM[:SS] [am/pm]            Explicit time.

They are implicitly added together to form a final datetime.

options:
  -h, --help [cmd]   Show this help message and exit
  -d, --db DB        Database file
```