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
        public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine, String msg, RecognitionException e) throws ParseCancellationException {
            throw new ParseCancellationException("Error");
        }

    }

    static class VariableUsage {

        private String variableName;
        private List<List<Token>> usage = new ArrayList<List<Token>>();

        public VariableUsage(String variableName) {
            this.variableName = variableName;
        }

        public void addContext(List<Token> context) {
            this.usage.add(context);
        }

        public String getVariableName() {
            return this.variableName;
        }

        public List<List<Token>> getContexts() {
            return this.usage;
        }

        public String toString() {
            String s = "Variable:" + this.variableName + "\n";
            for (List<Token> context: this.usage) {
                s += "[";
                for (Token tok: context) {
                    s += " '" + tok.getText() + "' ";
                }
                s += "]\n";
            }
            return s;
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

        private List<VariableUsage> usages = new ArrayList<VariableUsage>();
        private Stack<Integer> methodIdStack = new Stack<Integer>();
        private int lastNewMethodId = -1;
        private Map<Integer, Map<String, VariableUsage>> methodUsages = (
            new HashMap<Integer, Map<String, VariableUsage>>());

        private CommonTokenStream tokens;
        private int contextSize;

        public VariableWalker(CommonTokenStream tokens, int contextSize) {
            this.tokens = tokens;
            this.contextSize = contextSize;
        }

        private void startMethodScope() {
            // On entering a method, start listening for usages for just that method
            this.lastNewMethodId ++;
            int methodId = this.lastNewMethodId;
            methodIdStack.push(methodId);
            this.methodUsages.put(methodId, new HashMap<String, VariableUsage>());
        }

        private void endMethodScope() {
            // When we leave a method, this should stop being considered as the current
            // method for new usages found.
            int methodId = methodIdStack.pop();

            // Transfer all of the usages in this method into a central list.
            for (VariableUsage usage: this.methodUsages.get(methodId).values()) {
                this.usages.add(usage);
            }
        }

        public void enterLambdaExpression(JavaParser.LambdaExpressionContext ctx) {
            startMethodScope();
        }

        public void exitLambdaExpression(JavaParser.LambdaExpressionContext ctx) {
            endMethodScope();
        }

        public void enterMethodDeclaration(JavaParser.MethodDeclarationContext ctx) {
            startMethodScope();
        }

        public void exitMethodDeclaration(JavaParser.MethodDeclarationContext ctx) {
            endMethodScope();
        }

        private void saveIdentifierContext(ParserRuleContext ctx) {
            TerminalNode node = ctx.getToken(JavaLexer.IDENTIFIER, 0);
            if (node != null) {
                String variableName = ctx.getText();
                if (this.methodIdStack.size() > 0) {
                    int currentMethodId = this.methodIdStack.peek();
                    if (this.methodUsages.get(currentMethodId).get(variableName) != null) {
                        VariableUsage usage = this.methodUsages.get(currentMethodId).get(variableName);
                        int tokenIndex = ctx.getStart().getTokenIndex();
                        List<Token> context = this.tokens.get(tokenIndex - contextSize, tokenIndex + contextSize);
                        usage.addContext(context);
                    }
                }
            }
        }

        public void enterVariableDeclaratorId(JavaParser.VariableDeclaratorIdContext ctx) {
            if (this.methodIdStack.size() > 0) {

                int currentMethodId = this.methodIdStack.peek();

                // Create a new usage entry for this variable.
                String variableName = ctx.getText();
                this.methodUsages.get(currentMethodId).put(variableName, new VariableUsage(variableName));

                // Save this declaration as a usage.
                this.saveIdentifierContext(ctx);
            }
        }

        public void enterPrimary(JavaParser.PrimaryContext ctx) {
            // Whenever a variable is used, save this context.
            this.saveIdentifierContext(ctx);
        }

        public List<VariableUsage> getUsages() {
            return this.usages;
        }

        public void visitError() {
            System.out.println("Error");
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
                    VariableWalker listener = new VariableWalker(tokens, 5);
                    try {
                        walker.walk(listener, tree);
                    // Some of the files call a StackOverflow error.  I don't know why,
                    // but for now we're ignoring them and moving on.
                    } catch(StackOverflowError e) {
                        errorOccurred = true;
                        errorWriter.println("Stack Overflow for file: " + path.toString());
                    }
                    List<VariableUsage> usages = listener.getUsages();
                    for (VariableUsage usage: usages) {
                        String s = usage.getVariableName();
                        for (List<Token> context: usage.getContexts()) {
                            for (Token tok: context) {
                                s += "," + tok.getText();
                            }
                        }
                        writer.println(s);
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
            String outputFilename = "output.csv";
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
