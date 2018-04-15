## Getting Started

Before running this program, you need to fetch the
dependencies:

```bash
cd deps
source fetch_libs.sh
```

And then initialize the environment by running the script.
This adds the dependencies to the Java classpath.

```bash
cd ..  # back to the main directory
source init_env.sh
```

To extract contexts from Java files, first, compile the
extractor code:

```bash
javac ExtractContexts.java
```

Then run the extraction with this command:
```bash
java ExtractContexts <path-to-data-directory>
```

This program will search through the directory recursively
for all files with the extension `.java`, and will attempt
to extract contexts from each one.

The program outputs two files:

First, the contexts, in `output.csv`.  There is no header
for this file.  Each line starts with a variable name, and
is followed by a number of fixed-length contexts.  Each
context is just a sequence of token values, each one
separated by a comma.  All the contexts are included
back-to-back.  Note that some variables will have more (and
in some cases, many more) contexts than others.  If you need
all variables to have the same number of contexts, you must
do that in a post-processing step.

Second, it outputs an `errors.txt` file, which will list all
of the files that failed to parse.  You can get a sense for
about how many files failed to parse during extraction by
looking at the `.`s (successes) and `E`s (errors) reported
in the console output.

## Other configurations

Want to extract longer contexts?  Edit the `.java`
file---there's a variable for this.

## Regenerating the Java parser and base tree listener

If you want to change the Java grammar (`JavaParser.g4`,
`JavaLexer.g4`), you can regenerate the parser files with
this command:

```bash
antlr4 *.g4 -listener -o parser
```

This assumes you have already initialized the development
environment using `source init_env.sh`.
