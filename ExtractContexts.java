import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.FileVisitResult;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.DirectoryFileFilter;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.apache.commons.io.filefilter.TrueFileFilter;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import org.antlr.v4.runtime.misc.*;


public class ExtractContexts {

    static class ErrorListener extends BaseErrorListener {

        public static final ErrorListener INSTANCE = new ErrorListener();

        @Override
        public void syntaxError(
                Recognizer<?, ?> recognizer, Object offendingSymbol,
                int line, int charPositionInLine, String msg,
                RecognitionException e) throws ParseCancellationException {
            throw new ParseCancellationException("Error");
        }

    }

    static class VariableUsage {

        private String variableName;
        private List<List<String>> usage = new ArrayList<List<String>>();

        public VariableUsage(String variableName) {
            this.variableName = variableName;
        }

        public void addContext(List<String> context) {
            this.usage.add(context);
        }

        public String getVariableName() {
            return this.variableName;
        }

        public List<List<String>> getContexts() {
            return this.usage;
        }

        public String toString() {
            String s = "Variable:" + this.variableName + "\n";
            for (List<String> context: this.usage) {
                s += "[";
                for (String t: context) {
                    s += " '" + t + "' ";
                }
                s += "]\n";
            }
            return s;
        }

    }

    static class Terminal {

        public String text;
        public int scopeId;
        public boolean isVariableDeclaration;
        public boolean isVariableUse;

        public Terminal(String text, int scopeId,
                boolean isVariableDeclaration, boolean isVariableUse) {
            this.text = text;
            this.scopeId = scopeId;
            this.isVariableDeclaration = isVariableDeclaration;
            this.isVariableUse = isVariableUse;
        }

        public String toString() {
            String s = this.text + "(" + this.scopeId;
            if (this.isVariableDeclaration) s += "D";
            if (this.isVariableUse) s += "U";
            return s + ")";
        }
    }

    /*
     * Get usage context for all variables.
     * Currently limited to variables defined in a function's local scope, or
     * in a function's parameters---no class-level variables are tracked.
     * If a function is nested in another function and they use the same
     * variable, none of the nested function's uses will be counted for the
     * outer function.  Though this does visit nested functions.
     */
    static class VariableWalker extends JavaParserBaseListener {

        // Keep track of the scopes where each variable is found.
        private Stack<Integer> scopeIdStack;
        private Stack<String> scopeTypeStack;
        private int lastNewScopeId = -1;

        // Special tokens
        private final String PAD = "<<PAD>>";
        private final String REFERENCE = "<<REF>>";

        // Structures for saving unfinished usages:
        // List of recent tokens that have been seen:
        private List<Terminal> recentTerminals;
        // List of all contexts that are still being created:
        private Map<Integer, Map<String, List<List<Terminal>>>> unfinishedContexts = (
            new HashMap<Integer, Map<String, List<List<Terminal>>>>());
        // Table of usages, that maps from a scope and variable name to the usages:
        private Map<Integer, Map<String, VariableUsage>> usageTable = (
            new HashMap<Integer, Map<String, VariableUsage>>());

        // Other state for saving contexts.
        private CommonTokenStream tokens;
        private int contextSize;

        public VariableWalker(CommonTokenStream tokens, int contextSize) {

            this.tokens = tokens;
            this.contextSize = contextSize;

            // Pre-load pad tokens into the list of recent terminals.
            this.recentTerminals = new ArrayList<Terminal>();
            for (int i = 0; i < this.contextSize; i++) {
                this.recentTerminals.add(new Terminal(PAD, -1, false, false));
            }

            // Initialize stack with an invalid scope ID (so that we can peek
            // for some of the first terminals we encounter).
            this.scopeIdStack = new Stack<Integer>();
            this.scopeIdStack.push(this.lastNewScopeId);
            this.scopeTypeStack = new Stack<String>();
            this.scopeTypeStack.push("none");
        }

        private boolean isVariableUse(TerminalNode node) {
            ParseTree parentCtx = node.getParent();
            if (parentCtx instanceof JavaParser.PrimaryContext) {
                return true;
            }
            return false;
        }

        private boolean isVariableDeclaration(TerminalNode node) {
            ParseTree parentCtx = node.getParent();
            if (parentCtx instanceof JavaParser.VariableDeclaratorIdContext) {
                return true;
            }
            return false;
        }

        // As we're ending the walk, end all unterminated contexts, adding padding.
        public void exitCompilationUnit(JavaParser.CompilationUnitContext ctx) {
            updateUnfinishedContexts(null);
        }

        private List<String> finishContext(int scopeId, String variable, List<Terminal> terminals) {
            List<String> context = new ArrayList<String>();
            for (Terminal terminal: terminals) {
                if (terminal.text.equals(variable) &&
                    terminal.scopeId == scopeId &&
                    (terminal.isVariableUse || terminal.isVariableDeclaration)) {
                    context.add(REFERENCE);
                } else {
                    context.add(terminal.text);
                }
            }
            while (context.size() <= this.contextSize * 2 + 1) {
                context.add(PAD);
            }
            return context;
        }

        // Pass in "null" for the terminal if you want to flush all remaining unfinished contexts.
        private void updateUnfinishedContexts(Terminal terminal) {
            // TODO(andrewhead): Flush out the unfinished contexts at the end.
            // Each variable in each scope can have multiple contexts in progress.  Whenever one
            // of them finishes, we save it, and stop tracking it.
            for (int scopeId: this.unfinishedContexts.keySet()) {
                Map<String, List<List<Terminal>>> scopeContexts = this.unfinishedContexts.get(scopeId);
                for (String variable: scopeContexts.keySet()) {
                    List<List<Terminal>> variableContexts = scopeContexts.get(variable);
                    // Backwards iteration necessary so we can remove finished contexts.
                    for (int i = variableContexts.size() - 1; i >= 0; i--) {
                        List<Terminal> context = variableContexts.get(i);
                        if (terminal != null) {
                            context.add(terminal);
                        }
                        if (terminal == null || (context.size() >= this.contextSize * 2 + 1)) {
                            variableContexts.remove(context);
                            List<String> finishedContext = finishContext(scopeId, variable, context);
                            this.usageTable.get(scopeId).get(variable).addContext(finishedContext);
                        }
                    }
                }
            }
        }

        public void visitTerminal(TerminalNode node) {

            String text = node.getText();
            int scopeId = this.scopeIdStack.peek();
            boolean isVariableDeclaration = isVariableDeclaration(node);
            boolean isVariableUse = isVariableUse(node);

            // Create a new record for this terminal
            Terminal terminal = new Terminal(
                text, scopeId, isVariableDeclaration, isVariableUse);

            // Add to terminal to the fixed-length context buffer
            if (this.recentTerminals.size() >= this.contextSize) {
                this.recentTerminals.remove(0);
            }
            this.recentTerminals.add(terminal);

            // Add this terminal to any unfinished contexts
            updateUnfinishedContexts(terminal);

        }

        private void trackIdentifierContext(ParserRuleContext ctx) {
            TerminalNode node = ctx.getToken(JavaLexer.IDENTIFIER, 0);
            if (node != null) {
                String variableName = ctx.getText();
                int scopeId = this.scopeIdStack.peek();
                if (this.usageTable.get(scopeId) != null) {
                    Map<String, VariableUsage> scopeUsageTable = (this.usageTable.get(scopeId));
                    if (scopeUsageTable.get(variableName) != null) {
                        if (this.unfinishedContexts.get(scopeId) == null) {
                            this.unfinishedContexts.put(scopeId, new HashMap<String, List<List<Terminal>>>());
                        }
                        Map<String, List<List<Terminal>>> scopeContexts = this.unfinishedContexts.get(scopeId);
                        if (scopeContexts.get(variableName) == null) {
                            scopeContexts.put(variableName, new ArrayList<List<Terminal>>());
                        }
                        List<List<Terminal>> variableContexts = scopeContexts.get(variableName);
                        variableContexts.add(new ArrayList<Terminal>(this.recentTerminals));
                    }
                }
            }
        }


        public void enterVariableDeclaratorId(JavaParser.VariableDeclaratorIdContext ctx) {
            int scopeId = this.scopeIdStack.peek();
            String scopeType = this.scopeTypeStack.peek();

            // Only track variables that are defined at method level.
            if (!scopeType.equals("method")) return;

            // Create a new usage entry for this variable.
            String variableName = ctx.getText();
            this.usageTable.get(scopeId).put(variableName, new VariableUsage(variableName));

            // Save this declaration as a usage.
            this.trackIdentifierContext(ctx);
        }

        public void enterPrimary(JavaParser.PrimaryContext ctx) {
            // Whenever a variable is used, save this context.
            this.trackIdentifierContext(ctx);
        }

        // Utilities for starting and ending a scopeStarts a new scope whenever
        private void startScope(String scopeType) {
            // On entering a method, start listening for usages for just that method
            this.lastNewScopeId ++;
            int scopeId = this.lastNewScopeId;
            scopeIdStack.push(scopeId);
            this.usageTable.put(scopeId, new HashMap<String, VariableUsage>());
            scopeTypeStack.push(scopeType);
        }

        private void endScope() {
            scopeIdStack.pop();
            scopeTypeStack.pop();
        }

        public void enterClassBody(JavaParser.ClassBodyContext ctx) {
            startScope("class");
        }
        public void exitClassBody(JavaParser.ClassBodyContext ctx) {
            endScope();
        }
        public void enterLambdaExpression(JavaParser.LambdaExpressionContext ctx) {
            startScope("method");
        }
        public void exitLambdaExpression(JavaParser.LambdaExpressionContext ctx) {
            endScope();
        }
        public void enterConstructorDeclaration(JavaParser.ConstructorDeclarationContext ctx) {
            startScope("method");
        }
        public void exitConstructorDeclaration(JavaParser.ConstructorDeclarationContext ctx) {
            endScope();
        }
        public void enterMethodDeclaration(JavaParser.MethodDeclarationContext ctx) {
            startScope("method");
        }
        public void exitMethodDeclaration(JavaParser.MethodDeclarationContext ctx) {
            endScope();
        }

        public List<VariableUsage> getUsages() {
            List<VariableUsage> usages = new ArrayList<VariableUsage>();
            for (int scopeId: this.usageTable.keySet()) {
                Map<String, VariableUsage> scopeUsage = this.usageTable.get(scopeId);
                for (String variable: scopeUsage.keySet()) {
                    usages.add(scopeUsage.get(variable));
                }
            }
            return usages;
        }

    }

    static class ParseVisitor extends SimpleFileVisitor<Path> {

        PrintWriter writer;
        PrintWriter errorWriter;
        int numVisits = 0;

        public ParseVisitor(PrintWriter writer, PrintWriter errorWriter) {
            this.writer = writer;
            this.errorWriter = errorWriter;
        }

        @Override
        public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {

            // Only try to parse Java files
            if (!path.toString().endsWith(".java")) 
                return FileVisitResult.CONTINUE;

            // Parse the program
            CharStream input = null;
            try {
                input = CharStreams.fromFileName(path.toString());
            } catch (IOException e) {}

            boolean errorOccurred = false;
            numVisits += 1;

            if (input != null) { 
                CommonTokenStream tokens = null;
                JavaParser parser;
                ParseTree tree = null;
                try {
                    JavaLexer lexer = new JavaLexer(input);
                    lexer.removeErrorListeners();
                    lexer.addErrorListener(ErrorListener.INSTANCE);

                    tokens = new CommonTokenStream(lexer);
                    parser = new JavaParser(tokens);
                    parser.removeErrorListeners();
                    parser.addErrorListener(ErrorListener.INSTANCE);
                    tree = parser.compilationUnit();
                } catch (ParseCancellationException e) {
                    errorOccurred = true;
                    errorWriter.println("Failure to parse file: " + path.toString());
                }

                if (tree != null) {
                    // Fetch all variable contexts from the program
                    ParseTreeWalker walker = new ParseTreeWalker();
                    VariableWalker listener = new VariableWalker(tokens, 10);
                    try {
                        walker.walk(listener, tree);
                    // Some of the files call a StackOverflow error.  I don't know why,
                    // but for now we're ignoring them and moving on.
                    } catch(StackOverflowError e) {
                        errorOccurred = true;
                        errorWriter.println("Stack Overflow for file: " + path.toString());
                    }
                    List<VariableUsage> usages = listener.getUsages();
                    Gson gson = new GsonBuilder().disableHtmlEscaping().create();
                    for (VariableUsage usage: usages) {
                        String json = gson.toJson(usage);
                        writer.println(json);
                    }
                }

                if (errorOccurred) {
                    System.out.print("E");
                } else {
                    System.out.print(".");
                }

                if (numVisits % 100 == 0) {
                    System.out.print("(" + numVisits +")\n");
                }


            }
            return FileVisitResult.CONTINUE;
        }
    }

    public static void main(String[] args) {

        PrintWriter writer;
        PrintWriter errorWriter;
        try {
            String outputFilename = "output.json";
            String errorFilename = "errors.txt";
            writer = new PrintWriter(new File(outputFilename));
            errorWriter = new PrintWriter(new File(errorFilename));
        } catch (FileNotFoundException f) {
            System.out.println("Couldn't create output files");
            return;
        }

        String dirName = args[0];
        System.out.println("Now parsing programs");
        try {
            Files.walkFileTree(Paths.get(dirName), new ParseVisitor(writer, errorWriter));
        } catch (IOException e) {
            System.out.println("Couldn't finish tree walk:" + e);
        }

        System.out.println();
        writer.close();
        errorWriter.close();
    }

}
