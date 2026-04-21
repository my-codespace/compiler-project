/**
 * =============================================================================
 *  main.cpp  —  Math-to-Assembly Toy Compiler Front-End  (Interactive REPL)
 * =============================================================================
 *
 *  Three-stage pipeline:
 *
 *    [Source Text]
 *         │
 *         ▼  Stage 1 — Lexer
 *    [Token Stream]        std::string_view lexemes (zero heap allocation)
 *         │
 *         ▼  Stage 2 — Recursive-Descent Parser
 *    [AST]                 std::unique_ptr nodes   (automatic lifecycle)
 *         │
 *         ▼  Stage 3 — Code Generator  (Post-Order DFS)
 *    [ARM-like Pseudo-Assembly]
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *  Grammar (EBNF):
 *
 *    program    → statement EOF
 *    statement  → IDENT '=' expression        ← assignment (LL(2) lookahead)
 *               | expression
 *    expression → term  { ('+' | '-') term  } ← left-associative
 *    term       → unary { ('*' | '/') unary } ← left-associative, higher prec
 *    unary      → '-' unary | primary          ← right-recursive
 *    primary    → NUMBER | IDENT | '(' expression ')'
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *  Time & Space Complexity of the Recursive-Descent Parser
 *  (written for a competitive-programming audience):
 *
 *  Think of the token array as an input string of length n (n = #tokens).
 *
 *  Time  — O(n):
 *    Each grammar rule either consumes one token and returns, or loops over
 *    remaining tokens.  Because every token is consumed (advanced past) at
 *    most once across the entire parse, the total work is Σ O(1) over n
 *    tokens → O(n).  There is no backtracking; the grammar is LL(2), so the
 *    parser always knows which production to apply after looking ahead at most
 *    two tokens.
 *
 *  Space — O(n) for the AST heap + O(d) for the implicit call stack, where
 *    d = maximum nesting depth of the expression.
 *    • The AST has exactly one node per literal/identifier/operator → O(n).
 *    • The recursion depth is bounded by the nesting depth d of parentheses
 *      and unary operators.  Worst case (e.g., ((((x))))):  d = O(n).
 *      Typical well-structured input:  d = O(log n) for balanced trees.
 *
 *  Overall: O(n) time, O(n) space — optimal for a single-pass LL parser.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *  Build command:
 *    g++ -std=c++17 -Wall -Wextra -O2 -o compiler main.cpp
 *
 *  Standard  : C++17  (string_view, make_unique, if-init, structured bindings)
 *  Target ISA: ARM Cortex-M / PIC pseudo-assembly
 * =============================================================================
 */

#include <cassert>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// =============================================================================
//  §1  TOKENS
//      The atomic output of the lexer and the raw input of the parser.
// =============================================================================

/**
 * TokenKind — every syntactic category our language can produce.
 *
 * Stored as uint8_t so that sizeof(Token) stays small, improving cache
 * utilisation when the parser iterates over the token vector.
 */
enum class TokenKind : uint8_t {
    // ── Literals and names ──────────────────────────
    Number,     ///< integer literal      e.g. 42
    Ident,      ///< variable / name      e.g. x, result_val

    // ── Binary operators ────────────────────────────
    Plus,       ///< +
    Minus,      ///< -
    Star,       ///< *
    Slash,      ///< /
    Equals,     ///< = (assignment, not equality)

    // ── Grouping ────────────────────────────────────
    LParen,     ///< (
    RParen,     ///< )

    // ── Sentinel ────────────────────────────────────
    End,        ///< appended by the lexer to mark end-of-input
};

/// Human-readable label for a TokenKind — used in error messages.
constexpr std::string_view kindLabel(TokenKind k) noexcept {
    switch (k) {
        case TokenKind::Number:  return "Number";
        case TokenKind::Ident:   return "Ident";
        case TokenKind::Plus:    return "+";
        case TokenKind::Minus:   return "-";
        case TokenKind::Star:    return "*";
        case TokenKind::Slash:   return "/";
        case TokenKind::Equals:  return "=";
        case TokenKind::LParen:  return "(";
        case TokenKind::RParen:  return ")";
        case TokenKind::End:     return "<EOF>";
    }
    return "?"; // satisfies -Wreturn-type; unreachable
}

/**
 * Token — a (kind, lexeme, optional-value) triple.
 *
 *  lexeme  — std::string_view pointing DIRECTLY into the original source buffer.
 *            Zero allocation; zero copying.  Valid as long as the source string
 *            remains alive (it does throughout each compile() call).
 *  intVal  — pre-parsed integer value; meaningful only when kind == Number.
 */
struct Token {
    TokenKind        kind;
    std::string_view lexeme;   ///< zero-copy window into the source string
    int64_t          intVal{0};
};

// =============================================================================
//  §2  LEXER
//      Converts a raw source string_view into a flat std::vector<Token>.
//      Complexity: O(n) time, O(k) space  (k = number of tokens <= n).
// =============================================================================

/**
 * Lexer — single-pass, allocation-free tokeniser.
 *
 * The lexer stores only a std::string_view of the source, so it never copies
 * or owns the input text.  All Token::lexeme fields are windows into that
 * same buffer.
 *
 * Invariant after tokenize() returns:
 *   The last element of the returned vector always has kind == TokenKind::End.
 *   This sentinel lets the parser unconditionally index tokens[cursor + 1]
 *   without an out-of-bounds risk.
 */
class Lexer {
public:
    explicit Lexer(std::string_view src) noexcept
        : src_(src), pos_(0) {}

    /// Scan the entire source and return the complete token list.
    [[nodiscard]] std::vector<Token> tokenize() {
        std::vector<Token> out;
        out.reserve(src_.size() / 2 + 2); // avoids repeated reallocation

        while (pos_ < src_.size()) {
            skipWhitespace();
            if (pos_ >= src_.size()) break;

            const char c = src_[pos_];

            if (std::isdigit(static_cast<unsigned char>(c)))
                out.push_back(scanNumber());
            else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
                out.push_back(scanIdent());
            else
                out.push_back(scanSymbol()); // throws on unrecognised char
        }

        // Append the end-of-input sentinel (zero-length lexeme at pos_).
        out.push_back(Token{ TokenKind::End, src_.substr(pos_, 0), 0 });
        return out;
    }

private:
    std::string_view src_;
    std::size_t      pos_;

    void skipWhitespace() noexcept {
        while (pos_ < src_.size() &&
               std::isspace(static_cast<unsigned char>(src_[pos_])))
            ++pos_;
    }

    /// Scan a run of decimal digits and parse their integer value inline.
    Token scanNumber() noexcept {
        const std::size_t start = pos_;
        int64_t val = 0;
        while (pos_ < src_.size() &&
               std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            val = val * 10 + static_cast<int64_t>(src_[pos_] - '0');
            ++pos_;
        }
        return Token{ TokenKind::Number, src_.substr(start, pos_ - start), val };
    }

    /// Scan an identifier: [A-Za-z_][A-Za-z0-9_]*.
    Token scanIdent() noexcept {
        const std::size_t start = pos_;
        while (pos_ < src_.size() &&
               (std::isalnum(static_cast<unsigned char>(src_[pos_])) ||
                src_[pos_] == '_'))
            ++pos_;
        return Token{ TokenKind::Ident, src_.substr(start, pos_ - start), 0 };
    }

    /// Scan exactly one punctuation / operator character.
    Token scanSymbol() {
        const char        c     = src_[pos_];
        const std::size_t start = pos_++;

        TokenKind kind;
        switch (c) {
            case '+': kind = TokenKind::Plus;   break;
            case '-': kind = TokenKind::Minus;  break;
            case '*': kind = TokenKind::Star;   break;
            case '/': kind = TokenKind::Slash;  break;
            case '=': kind = TokenKind::Equals; break;
            case '(': kind = TokenKind::LParen; break;
            case ')': kind = TokenKind::RParen; break;
            default:
                throw std::runtime_error(
                    std::string("Lexer: unexpected character '") + c + '\'');
        }
        return Token{ kind, src_.substr(start, 1), 0 };
    }
};

// =============================================================================
//  §3  ABSTRACT SYNTAX TREE (AST)
// =============================================================================

// ── 3a. Node kind tag ────────────────────────────────────────────────────────

/**
 * NodeKind — an enum tag stored in every ASTNode.
 *
 * The code generator switches on this tag and uses static_cast<> to obtain the
 * concrete type.  This avoids RTTI / dynamic_cast overhead and mirrors the
 * design of production compilers (Clang: Stmt::StmtClass, LLVM: Value::ValueID,
 * GCC: tree_code).
 */
enum class NodeKind : uint8_t {
    Number,      ///< integer literal leaf
    Variable,    ///< variable reference leaf
    BinaryOp,    ///< binary arithmetic interior node
    Assign,      ///< assignment statement interior node
    UnaryMinus,  ///< unary negation interior node
};

// Forward-declare the alias used throughout all node structs.
using NodePtr = std::unique_ptr<struct ASTNode>;

// ── 3b. Base class ───────────────────────────────────────────────────────────

/**
 * ASTNode — polymorphic base for every node in the syntax tree.
 *
 * Memory ownership (RAII via unique_ptr):
 *   Every parent node owns its children through std::unique_ptr<ASTNode>.
 *   When the root goes out of scope, the entire tree is recursively destroyed
 *   in O(n) destructor calls — zero manual `delete`, zero memory leaks.
 *
 * Non-copyable: tree nodes are uniquely owned; copy constructor and assignment
 * are explicitly deleted to prevent accidental shared ownership.
 */
struct ASTNode {
    const NodeKind kind;

    explicit ASTNode(NodeKind k) noexcept : kind(k) {}
    virtual ~ASTNode() = default;

    ASTNode(const ASTNode&)            = delete;
    ASTNode& operator=(const ASTNode&) = delete;

    /// Return an indented, human-readable representation of the subtree.
    /// @param depth  current indentation level (each level = 2 spaces)
    [[nodiscard]] virtual std::string dump(int depth) const = 0;

protected:
    static std::string pad(int depth) { return std::string(depth * 2, ' '); }
};

// ── 3c. Concrete node types ──────────────────────────────────────────────────

/// Leaf: an integer constant  e.g. 42
struct NumberNode final : ASTNode {
    int64_t value;
    explicit NumberNode(int64_t v) : ASTNode(NodeKind::Number), value(v) {}
    [[nodiscard]] std::string dump(int depth) const override {
        return pad(depth) + "NumberLiteral(" + std::to_string(value) + ')';
    }
};

/// Leaf: a variable reference  e.g. x, result
struct VariableNode final : ASTNode {
    std::string name; // owning std::string — safe after source string goes away
    explicit VariableNode(std::string_view n)
        : ASTNode(NodeKind::Variable), name(n) {}
    [[nodiscard]] std::string dump(int depth) const override {
        return pad(depth) + "VarRef(" + name + ')';
    }
};

/**
 * Interior: a binary arithmetic operation.
 * `op` is one of '+', '-', '*', '/'.
 *
 * Tree shape for  a + b:
 *     BinaryOp('+')
 *     ├── lhs  (sub-expression a)
 *     └── rhs  (sub-expression b)
 */
struct BinaryOpNode final : ASTNode {
    char    op;
    NodePtr lhs;
    NodePtr rhs;
    BinaryOpNode(char op, NodePtr l, NodePtr r)
        : ASTNode(NodeKind::BinaryOp),
          op(op), lhs(std::move(l)), rhs(std::move(r)) {}
    [[nodiscard]] std::string dump(int depth) const override {
        return pad(depth) + std::string("BinaryOp('") + op + "')\n" +
               lhs->dump(depth + 1) + '\n' +
               rhs->dump(depth + 1);
    }
};

/**
 * Interior: a variable assignment statement  IDENT = expr.
 *     Assign(x)
 *     └── rhs  (expression whose result will be stored in x)
 */
struct AssignNode final : ASTNode {
    std::string name;
    NodePtr     rhs;
    AssignNode(std::string_view n, NodePtr r)
        : ASTNode(NodeKind::Assign), name(n), rhs(std::move(r)) {}
    [[nodiscard]] std::string dump(int depth) const override {
        return pad(depth) + "Assign(" + name + ")\n" +
               rhs->dump(depth + 1);
    }
};

/**
 * Interior: unary negation  -expr.
 *     UnaryMinus
 *     └── operand  (sub-expression to negate)
 */
struct UnaryMinusNode final : ASTNode {
    NodePtr operand;
    explicit UnaryMinusNode(NodePtr op)
        : ASTNode(NodeKind::UnaryMinus), operand(std::move(op)) {}
    [[nodiscard]] std::string dump(int depth) const override {
        return pad(depth) + "UnaryMinus\n" +
               operand->dump(depth + 1);
    }
};

// =============================================================================
//  §4  RECURSIVE-DESCENT PARSER
//      Transforms a token stream into an AST.
// =============================================================================

/**
 * Parser — LL(2) recursive-descent parser.
 *
 * Each grammar non-terminal is a private method.  Operator precedence is
 * encoded structurally: higher-precedence operators are handled by methods
 * invoked DEEPER in the call chain.
 *
 * Precedence table (low -> high):
 *   parseStatement  →  = assignment
 *   parseExpression →  + -
 *   parseTerm       →  * /
 *   parseUnary      →  unary -
 *   parsePrimary    →  literals · identifiers · ( ) grouping
 *
 * Complexity:
 *   Time  O(n)  — each token consumed at most once, no backtracking.
 *   Space O(d)  — implicit call stack = nesting depth d.
 *         O(n)  — AST nodes on the heap.
 */
class Parser {
public:
    explicit Parser(std::vector<Token> tokens)
        : tokens_(std::move(tokens)), cursor_(0) {}

    [[nodiscard]] NodePtr parseProgram() {
        NodePtr root = parseStatement();
        expect(TokenKind::End);
        return root;
    }

private:
    std::vector<Token> tokens_;
    std::size_t        cursor_;

    [[nodiscard]] const Token& peek()     const noexcept { return tokens_[cursor_];     }
    [[nodiscard]] const Token& peekNext() const noexcept { return tokens_[cursor_ + 1]; }
    const Token& advance() noexcept { return tokens_[cursor_++]; }

    const Token& expect(TokenKind expected) {
        if (peek().kind != expected)
            throw std::runtime_error(
                "Parser: expected '" + std::string(kindLabel(expected)) +
                "' but got '"        + std::string(kindLabel(peek().kind)) +
                "' (text: \""        + std::string(peek().lexeme) + "\")");
        return advance();
    }

    /**
     * statement → IDENT '=' expression  |  expression
     *
     * LL(2) lookahead: if the current token is IDENT and the NEXT is '=',
     * take the assignment branch.  Otherwise fall into expression.
     * Both peek() and peekNext() are O(1) array accesses.
     */
    NodePtr parseStatement() {
        if (peek().kind     == TokenKind::Ident &&
            peekNext().kind == TokenKind::Equals) {
            std::string varName{ peek().lexeme };
            advance(); // consume IDENT
            advance(); // consume '='
            NodePtr rhs = parseExpression();
            return std::make_unique<AssignNode>(varName, std::move(rhs));
        }
        return parseExpression();
    }

    /**
     * expression → term { ('+' | '-') term }
     * Left-associative: the while-loop folds left, building a left-leaning tree.
     */
    NodePtr parseExpression() {
        NodePtr node = parseTerm();
        while (peek().kind == TokenKind::Plus ||
               peek().kind == TokenKind::Minus) {
            const char op = (advance().kind == TokenKind::Plus) ? '+' : '-';
            NodePtr rhs   = parseTerm();
            node = std::make_unique<BinaryOpNode>(op, std::move(node), std::move(rhs));
        }
        return node;
    }

    /**
     * term → unary { ('*' | '/') unary }
     * Higher precedence than expression by virtue of being called from it.
     */
    NodePtr parseTerm() {
        NodePtr node = parseUnary();
        while (peek().kind == TokenKind::Star ||
               peek().kind == TokenKind::Slash) {
            const char op = (advance().kind == TokenKind::Star) ? '*' : '/';
            NodePtr rhs   = parseUnary();
            node = std::make_unique<BinaryOpNode>(op, std::move(node), std::move(rhs));
        }
        return node;
    }

    /**
     * unary → '-' unary | primary
     * Right-recursive: ---x parses as -(-(-(x))).
     */
    NodePtr parseUnary() {
        if (peek().kind == TokenKind::Minus) {
            advance();
            return std::make_unique<UnaryMinusNode>(parseUnary());
        }
        return parsePrimary();
    }

    /**
     * primary → NUMBER | IDENT | '(' expression ')'
     * Base case; parentheses re-enter parseExpression resetting precedence.
     */
    NodePtr parsePrimary() {
        const Token& tok = peek();

        if (tok.kind == TokenKind::Number) {
            advance();
            return std::make_unique<NumberNode>(tok.intVal);
        }
        if (tok.kind == TokenKind::Ident) {
            advance();
            return std::make_unique<VariableNode>(tok.lexeme);
        }
        if (tok.kind == TokenKind::LParen) {
            advance();
            NodePtr inner = parseExpression();
            expect(TokenKind::RParen);
            return inner;
        }

        throw std::runtime_error(
            "Parser: unexpected token '" + std::string(tok.lexeme) +
            "' (kind: " + std::string(kindLabel(tok.kind)) + ")");
    }
};

// =============================================================================
//  §5  CODE GENERATOR
//      Traverses the AST in post-order DFS and emits ARM-like pseudo-assembly.
// =============================================================================

/**
 * CodeGenerator — AST -> ARM Pseudo-Assembly via post-order depth-first search.
 *
 * Virtual Register Model (SSA-style):
 *   Registers R0, R1, R2, ... — each written exactly once, then only read.
 *   regCounter_ tracks the next free index.
 *   Each emit*() returns the register index holding its result.
 *
 * Instruction Set (ARM Cortex-M / PIC-inspired):
 *   LDR  Rd, #imm     Rd <- immediate constant
 *   LDR  Rd, [var]    Rd <- memory[var]
 *   STR  Rs, [var]    memory[var] <- Rs
 *   NEG  Rd, Rm       Rd <- 0 - Rm  (unary negate)
 *   ADD  Rd, Ra, Rb   Rd <- Ra + Rb
 *   SUB  Rd, Ra, Rb   Rd <- Ra - Rb
 *   MUL  Rd, Ra, Rb   Rd <- Ra * Rb
 *   SDIV Rd, Ra, Rb   Rd <- Ra / Rb  (signed integer divide)
 *
 * Why post-order?
 *   The parent instruction is always emitted AFTER both operand sub-trees,
 *   guaranteeing every instruction's inputs are produced before it runs —
 *   the topological ordering required by any sequential instruction stream.
 */
class CodeGenerator {
public:
    [[nodiscard]] std::string generate(const ASTNode* root) {
        regCounter_ = 0;
        listing_.clear();
        comment("---- begin codegen  (post-order DFS) ----");
        const int resultReg = emitNode(root);
        comment("---- result in " + regName(resultReg) + " ----");
        return listing_;
    }

private:
    int         regCounter_{0};
    std::string listing_;

    int  allocReg() noexcept { return regCounter_++; }
    static std::string regName(int r) { return 'R' + std::to_string(r); }

    void instr(const std::string& text) {
        listing_ += "    "; listing_ += text; listing_ += '\n';
    }
    void comment(const std::string& text) {
        listing_ += "  ; "; listing_ += text; listing_ += '\n';
    }

    /**
     * Central dispatch via NodeKind tag + static_cast.
     * Avoids RTTI / dynamic_cast overhead — same pattern as Clang and LLVM.
     */
    int emitNode(const ASTNode* node) {
        switch (node->kind) {
            case NodeKind::Number:
                return emitNumber(static_cast<const NumberNode*>(node));
            case NodeKind::Variable:
                return emitVariable(static_cast<const VariableNode*>(node));
            case NodeKind::BinaryOp:
                return emitBinaryOp(static_cast<const BinaryOpNode*>(node));
            case NodeKind::Assign:
                return emitAssign(static_cast<const AssignNode*>(node));
            case NodeKind::UnaryMinus:
                return emitUnaryMinus(static_cast<const UnaryMinusNode*>(node));
        }
        throw std::logic_error("CodeGen: unhandled NodeKind in emitNode()");
    }

    int emitNumber(const NumberNode* n) {
        const int rd = allocReg();
        instr("LDR  " + regName(rd) + ", #" + std::to_string(n->value));
        return rd;
    }

    int emitVariable(const VariableNode* n) {
        const int rd = allocReg();
        instr("LDR  " + regName(rd) + ", [" + n->name + ']');
        return rd;
    }

    /**
     * Post-order for BinaryOpNode:
     *   Phase 1: recurse lhs  → Ra   (all lhs instructions come first)
     *   Phase 2: recurse rhs  → Rb   (all rhs instructions come next)
     *   Phase 3: emit OP Rc, Ra, Rb  (parent always last)
     */
    int emitBinaryOp(const BinaryOpNode* n) {
        comment(std::string("eval LHS of '") + n->op + '\'');
        const int ra = emitNode(n->lhs.get());
        comment(std::string("eval RHS of '") + n->op + '\'');
        const int rb = emitNode(n->rhs.get());
        const int rd = allocReg();

        const char* mnem = nullptr;
        switch (n->op) {
            case '+': mnem = "ADD "; break;
            case '-': mnem = "SUB "; break;
            case '*': mnem = "MUL "; break;
            case '/': mnem = "SDIV"; break;
            default:
                throw std::logic_error(
                    std::string("CodeGen: unknown operator '") + n->op + '\'');
        }
        instr(std::string(mnem) + ' ' +
              regName(rd) + ", " + regName(ra) + ", " + regName(rb));
        return rd;
    }

    int emitAssign(const AssignNode* n) {
        comment("evaluating RHS for assignment -> [" + n->name + ']');
        const int rs = emitNode(n->rhs.get());
        instr("STR  " + regName(rs) + ", [" + n->name + ']');
        comment("[" + n->name + "] written");
        return rs;
    }

    int emitUnaryMinus(const UnaryMinusNode* n) {
        comment("eval operand of unary '-'");
        const int ra = emitNode(n->operand.get());
        const int rd = allocReg();
        instr("NEG  " + regName(rd) + ", " + regName(ra));
        return rd;
    }
};

// =============================================================================
//  §6  UTILITIES
// =============================================================================

/**
 * trimmed() — strip leading and trailing whitespace (including '\r') from sv.
 *
 * Why this matters for REPL robustness:
 *   1. Trailing spaces cause the parser's End-sentinel check to fail with a
 *      confusing "unexpected token" error instead of clean EOF.
 *   2. '\r' appears when input is pasted from Windows-formatted text
 *      (CRLF line endings) on a Unix terminal.
 *
 * Returns a string_view window into the same buffer — zero allocation.
 */
static std::string_view trimmed(std::string_view sv) noexcept {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front())))
        sv.remove_prefix(1);
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back())))
        sv.remove_suffix(1);
    return sv;
}

// =============================================================================
//  §7  DRIVER — wires the three stages together
// =============================================================================

/**
 * compile() — run the full Lex -> Parse -> CodeGen pipeline on one source line.
 *
 * stdout carries the assembly listing.
 * stderr carries prompts and error messages.
 *
 * Keeping them separate means the user can redirect stdout to capture a clean
 * assembly file without REPL chrome:
 *   ./compiler > out.asm
 */
static void compile(std::string_view source) {
    const std::string divider(56, '=');
    std::cout << '\n' << divider << '\n';
    std::cout << "  SOURCE:  " << source << '\n';
    std::cout << divider << '\n';

    try {
        // ── Stage 1: Lex ─────────────────────────────────────────────────────
        Lexer              lexer(source);
        std::vector<Token> tokens = lexer.tokenize();

        std::cout << "\n[Stage 1 - Token Stream]\n";
        for (const auto& tok : tokens) {
            std::cout << "    " << kindLabel(tok.kind);
            if (tok.kind == TokenKind::Number)
                std::cout << "(" << tok.intVal << ")";
            else if (tok.kind == TokenKind::Ident)
                std::cout << "(\"" << tok.lexeme << "\")";
            std::cout << '\n';
        }

        // ── Stage 2: Parse ────────────────────────────────────────────────────
        Parser  parser(std::move(tokens));
        NodePtr ast = parser.parseProgram();

        std::cout << "\n[Stage 2 - Abstract Syntax Tree]\n";
        std::cout << ast->dump(1) << '\n';

        // ── Stage 3: Code Generation ──────────────────────────────────────────
        CodeGenerator codegen;
        std::string   listing = codegen.generate(ast.get());

        std::cout << "\n[Stage 3 - ARM Pseudo-Assembly]\n";
        std::cout << listing;

        // `ast` falls out of scope -> unique_ptr cascade-deletes the entire tree
        // via the virtual destructor chain.  O(n) destruction, zero leaks.

    } catch (const std::exception& ex) {
        std::cerr << "  [COMPILE ERROR]  " << ex.what() << '\n';
    }
}

// =============================================================================
//  §8  MAIN — Interactive REPL
// =============================================================================

/// Print the startup banner once on launch.
static void printBanner() {
    std::cout
        << "============================================================\n"
        << "  Math-to-Assembly Toy Compiler Front-End\n"
        << "  Standard  : C++17  |  Target: ARM Pseudo-Assembly\n"
        << "  Commands  : 'help' for examples  |  'exit' / 'quit' to stop\n"
        << "  Tip       : stdout = assembly listing (safe to redirect)\n"
        << "              prompts and errors go to stderr\n"
        << "============================================================\n";
}

/// Show four example expressions that cover every language feature.
static void printHelp() {
    std::cout
        << "\n  Example expressions:\n"
        << "    3 + 5                          basic arithmetic\n"
        << "    x = 5 + 3 * 8                 assignment + precedence\n"
        << "    result = (a + b) * (a - b)     variables + grouping\n"
        << "    y = -(x + 2) * 4 / (z - 1)    unary minus + all operators\n\n";
}

int main() {
    printBanner();

    std::string input;

    while (true) {
        // Prompt to stderr — does not appear in redirected stdout.
        std::cerr << "\n>> ";

        // getline() returns false on EOF (Ctrl+D on Linux, Ctrl+Z on Windows).
        if (!std::getline(std::cin, input)) break;

        // C++17 if-with-initialiser: `expr` is scoped to this block only.
        // trimmed() strips whitespace and '\r' before any further processing.
        if (const auto expr = trimmed(input); expr.empty()) {
            continue;                          // blank line — prompt again
        } else if (expr == "exit" || expr == "quit") {
            break;
        } else if (expr == "help") {
            printHelp();
        } else {
            compile(expr);                     // run the full pipeline
        }
    }

    std::cerr << "\n  Goodbye.\n";
    return 0;
}

/*
 * =============================================================================
 *  COMPLEXITY ANALYSIS — for a competitive-programming audience
 * =============================================================================
 *
 *  Input size:  n = number of tokens produced by the lexer
 *               (proportional to the number of characters in the source).
 *
 *  ── Lexer ────────────────────────────────────────────────────────────────
 *  Time  O(n)   — one character at a time, no look-back.
 *  Space O(k)   — k tokens stored in the output vector; k <= n.
 *                  No source copying (string_view windows).
 *
 *  ── Parser ───────────────────────────────────────────────────────────────
 *  Time  O(n):
 *    Each token is consumed by exactly one advance() call across the entire
 *    recursion.  Even though there are O(n) recursive calls in the worst
 *    case, their work is amortised: the cursor only moves forward, so the
 *    total number of token accesses is O(n).  This is the classic argument
 *    for why recursive-descent parsers on LL grammars run in linear time —
 *    it is equivalent to showing that a pushdown automaton for an LL(k)
 *    grammar runs in O(n) steps.
 *
 *  Space O(n) heap (AST nodes) + O(d) stack (recursion depth):
 *    The AST has exactly one node per leaf (literal / identifier) and one
 *    per operator, so the node count <= 2n - 1 = O(n).
 *    The recursive call stack depth equals the expression's nesting depth d.
 *    Worst case: ((((x)))) gives d = O(n).
 *    Balanced, typical expressions give d = O(log n).
 *
 *  ── Code Generator ───────────────────────────────────────────────────────
 *  Time  O(n)   — each AST node visited once in the DFS.
 *  Space O(n)   — one virtual register allocated per node.
 *         O(d)  — DFS recursion stack (same bound as the parser).
 *
 *  ── Overall pipeline ─────────────────────────────────────────────────────
 *  Time  O(n)   — all three stages are linear in the input size.
 *  Space O(n)   — dominated by the AST node storage.
 * =============================================================================
 */