"""Safe AST and token parsing for Python source files and diff hunks."""

import ast
import io
import textwrap
import tokenize
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


_IGNORED_TOKEN_TYPES = {
    tokenize.COMMENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.INDENT,
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.ENDMARKER,
}


@dataclass(frozen=True)
class SyntaxDiagnostic:
    """Represents a syntax or indentation error during parsing."""
    line_number: int
    message: str
    text: str


@dataclass
class ParsedAST:
    """Encapsulates the parsed AST and line mapping metadata."""
    tree: ast.AST
    line_offset: int = 0
    diagnostic: Optional[SyntaxDiagnostic] = None


def get_call_name(node: ast.AST) -> Optional[str]:
    """Extract a dotted or simple name from an ast.Call or expression.
    
    Examples:
        eval -> "eval"
        os.system -> "os.system"
        subprocess.run -> "subprocess.run"
        cursor.execute -> "cursor.execute"
        hashlib.md5 -> "hashlib.md5"
    """
    if isinstance(node, ast.Call):
        func = node.func
    else:
        func = node

    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        value_name = get_call_name(func.value)
        if value_name:
            return f"{value_name}.{func.attr}"
        return func.attr
    return None


def get_constant_value(node: ast.AST) -> Optional[Any]:
    """Safely extract constant string/int/bool values from AST."""
    if isinstance(node, ast.Constant):
        return node.value
    return None


def tokenize_source(source: str) -> List[tokenize.TokenInfo]:
    """Generate tokens from source string while filtering trivia (comments, indents, newlines, endmarker)."""
    result: List[tokenize.TokenInfo] = []
    generator = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        while True:
            result.append(next(generator))
    except StopIteration:
        pass
    except (IndentationError, tokenize.TokenError):
        pass
    return [token for token in result if token.type not in _IGNORED_TOKEN_TYPES]


def parse_source_or_fragment(lines_text: Sequence[str]) -> Tuple[Optional[ast.AST], int, Optional[SyntaxDiagnostic]]:
    """Safely parse code lines that may be a complete file or an indented diff hunk.
    
    Returns:
        (tree, line_offset, syntax_diagnostic)
        where line_offset is the line number added by wrappers (0 if no wrapper).
    """
    raw_source = "\n".join(lines_text)
    if not raw_source.strip():
        return None, 0, None

    # 1. Try direct parsing
    try:
        tree = ast.parse(raw_source)
        return tree, 0, None
    except (SyntaxError, IndentationError):
        pass

    # 2. Try dedented parsing
    dedented = textwrap.dedent(raw_source)
    try:
        tree = ast.parse(dedented)
        return tree, 0, None
    except (SyntaxError, IndentationError):
        pass

    # 3. Try wrapping inside a function body (handles returns, yields, statements)
    indented_for_func = textwrap.indent(dedented, "    ")
    func_wrapper = f"def _dummy_container():\n{indented_for_func}"
    try:
        tree = ast.parse(func_wrapper)
        return tree, 1, None
    except (SyntaxError, IndentationError):
        pass

    # 4. Try wrapping inside a try-block (handles bare except or except handlers)
    try_wrapper = f"try:\n    pass\n{dedented}"
    try:
        tree = ast.parse(try_wrapper)
        return tree, 2, None
    except (SyntaxError, IndentationError):
        pass

    # 5. Try wrapping inside an if-block (handles elif / else)
    if_wrapper = f"if True:\n    pass\n{dedented}"
    try:
        tree = ast.parse(if_wrapper)
        return tree, 2, None
    except (SyntaxError, IndentationError) as err:
        err_lineno = getattr(err, "lineno", 1) or 1
        err_msg = getattr(err, "msg", "Invalid syntax")
        err_text = getattr(err, "text", "") or ""
        return None, 0, SyntaxDiagnostic(line_number=err_lineno, message=err_msg, text=err_text.strip())
