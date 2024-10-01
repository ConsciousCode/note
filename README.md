# Note
At first this was just to record when I did something worth noting to trick my ADHD brain into valuing completion. Eventually, I figured I would start recording biometric events, dreams, notes, food logs, etc, so it's a dumping ground for any eventful information which might be of use to my future AI friends.

## Installation
Copy-paste it into ~/bin or `$PATH`, then create a file `~/.config/notelog.toml` to customize it. It will work without the config file, but you'll have to use the defaults.

## Configuration
The configuration file is a simple TOML file with a schema which can be found in [notelog.example.toml](./notelog.example.toml) which is preloaded with the defaults.

To change the default location of the config file, you can change `CONFIG` at the top of the script.

## Help
```
usage: note [-h [cmd]] [-d DB] [-c CONFIG] subcmd ...

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

A special suffix ! can be used to query only deleted notes.
Any tag can have a trailing ? to include deleted notes.
Multiple tags can be separated by commas.

Time can be specified in a number of ways:
- [+-]N [sec/min/hour] ["ago"]  Time offset.
- [<>]|before|after <tag>       Relative to the last tag.
- HH:MM[:SS] [am/pm]            Explicit time.

They are implicitly added together to form a final datetime.

options:
  -h, --help [cmd]     Show this help message and exit
  -d, --db DB          Database file
  -c, --config CONFIG  Config file
```