# C++ Math-to-Assembly Compiler Front-End

A custom, single-pass compiler front-end built entirely from scratch in modern C++17. This project takes raw mathematical strings, performs lexical analysis and recursive-descent parsing, and generates valid ARM-style pseudo-assembly instructions utilizing a virtual register model.

Built as a demonstration of low-level systems programming, memory management, and tree-traversal algorithms.

## 🛠️ Technical Architecture

The pipeline is split into three distinct stages, relying heavily on zero-allocation data structures and modern C++ memory safety (RAII):

1. **The Lexer (Lexical Analysis):** Scans raw input into a flat token stream in `O(n)` time. Utilizes `std::string_view` to create zero-copy windows directly into the source buffer, preventing unnecessary heap allocations.
2. **The Parser (Abstract Syntax Tree):** An LL(2) Recursive-Descent parser that structuralizes the token stream into an AST. Operator precedence (e.g., multiplication before addition) is naturally enforced via the call stack. Memory lifecycle is strictly managed via `std::unique_ptr` cascade-deletions to ensure zero memory leaks.
3. **The Code Generator:** Traverses the AST using a Post-Order Depth-First Search (DFS). This topological sort ensures operand instructions are generated before the operations that consume them, outputting Static Single-Assignment (SSA) style ARM assembly (e.g., `LDR`, `STR`, `MUL`, `ADD`).

## 🚀 How to Run

Compile the source using GCC/G++ (Requires C++17 or higher):

```bash
g++ -std=c++17 -Wall -Wextra -O2 -o compiler main.cpp
