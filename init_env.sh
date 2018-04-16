#! /bin/bash

export CLASSPATH=.:$CLASSPATH:deps/commons-io-2.6.jar
export CLASSPATH=$CLASSPATH:deps/antlr-4.7.1-complete.jar
export CLASSPATH=$CLASSPATH:deps/gson-2.8.2.jar
export CLASSPATH=$CLASSPATH:parser/

alias antlr4='java -jar deps/antlr-4.7.1-complete.jar'
alias grun='java org.antlr.v4.gui.TestRig'
