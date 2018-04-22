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

First, the contexts, in `output.json`.  Each line includes
a separate JSON object, with a `variableName` (the actual
variable name) and a `usage` object, which is an array of
contexts of tokens.  Some of these tokens will be `<<PAD>>`,
(a padding token if the file ran out of characters), or
`<<REF>>`, which means that that token is a reference to
the variable whose name is in `variableName`.  Some
variables will have more contexts than others.

Second, it outputs an `errors.txt` file, which will list all
of the files that failed to parse.  You can get a sense for
about how many files failed to parse during extraction by
looking at the `.`s (successes) and `E`s (errors) reported
in the console output.

## Other configurations

Want to extract longer contexts?  Edit the `.java`
file---there's a variable for this.

There is also a switch for obfuscating variables in the
`.java` file called `OBFUSCATE`.  Set this to true or false
based on whether you want the dataset to include variable
names.

## Regenerating the Java parser and base tree listener

If you want to change the Java grammar (`JavaParser.g4`,
`JavaLexer.g4`), you can regenerate the parser files with
this command:

```bash
antlr4 *.g4 -listener -o parser
```

This assumes you have already initialized the development
environment using `source init_env.sh`.
