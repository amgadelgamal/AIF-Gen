import ast
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

# Adjust this to your project's package prefix (i.e. the part that appears on disk above your base_dir)
PACKAGE_PREFIX = 'benchmarks.COPR.'


def get_module_name(file_path: str, base_dir: str) -> str:
    # Convert file path relative to base_dir to a module name with dots.
    rel_path = os.path.relpath(file_path, base_dir)
    if rel_path.endswith('.py'):
        rel_path = rel_path[:-3]  # remove .py extension
    # Normalize __init__.py: use the directory as module name.
    if os.path.basename(file_path) == '__init__.py':
        rel_path = os.path.dirname(rel_path)
    return '.'.join(Path(rel_path).parts) if rel_path else ''


def resolve_relative_import(
    this_module: Optional[str], level: int, imported_module: Optional[str]
) -> Optional[str]:
    # For a relative import in this_module, compute the absolute module name.
    if this_module is None:
        return None
    parts = this_module.split('.')
    if level > len(parts):
        return imported_module or ''
    base = parts[:-level]
    if imported_module:
        base.append(imported_module)
    resolved_module = '.'.join(base)
    return resolved_module


def find_irrelevant_files(
    algorithm_name: str, base_dir: str = 'benchmarks/COPR'
) -> dict:
    """Analyze all Python files under base_dir. Identify initial files whose filename contains
    'algorithm_name' and then do a recursive import analysis to determine the transitively used files.
    Return a dict with keys:
      'relevant_files' - set of paths that are transitively required
      'irrelevant_files' - set of paths under base_dir not reached.
    """
    algorithm_name = algorithm_name.lower()
    base_dir = os.path.abspath(base_dir)
    all_files = []
    file_to_module = {}
    module_to_file = {}
    # Also track Python files directly
    path_cache = {}

    # Collect all *.py files and build a mapping from file to module name.
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
                mod_name = get_module_name(file_path, base_dir)
                file_to_module[file_path] = mod_name
                if mod_name:
                    module_to_file[mod_name] = file_path

                # Store the path without .py extension for direct lookup
                path_without_ext = file_path[:-3]
                path_cache[os.path.basename(path_without_ext)] = file_path

    # Initial files are those whose filename (or full path) contains the algorithm name.
    initial_files = [
        f for f in all_files if algorithm_name in os.path.basename(f).lower()
    ]
    if not initial_files:
        print(f"No files found with '{algorithm_name}' in their name under {base_dir}")
        return {'relevant_files': set(), 'irrelevant_files': set(all_files)}

    dependency_graph = defaultdict(set)
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
            this_module = file_to_module.get(file_path, '')
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    dep_module = None
                    if node.level > 0:
                        dep_module = resolve_relative_import(
                            this_module, node.level, node.module
                        )
                    else:
                        dep_module = node.module

                    if not dep_module:
                        continue

                    # If the import uses the full package prefix, strip it.
                    if dep_module.startswith(PACKAGE_PREFIX):
                        dep_module = dep_module[len(PACKAGE_PREFIX) :]
                    elif dep_module.startswith('copr.'):
                        dep_module = dep_module[len('copr.') :]

                    # For each name being imported
                    for alias in node.names:
                        import_name = alias.name
                        full_import = (
                            f'{dep_module}.{import_name}' if dep_module else import_name
                        )

                        # Try direct module name match first
                        if full_import in module_to_file:
                            dependency_graph[file_path].add(module_to_file[full_import])
                        # Try to find the exact file
                        elif import_name in path_cache:
                            # Try to find the file in the right package
                            pkg_path = os.path.join(base_dir, *dep_module.split('.'))
                            target_path = os.path.join(pkg_path, f'{import_name}.py')
                            if os.path.exists(target_path):
                                dependency_graph[file_path].add(target_path)
                            else:
                                # Last resort - just use the first match found
                                dependency_graph[file_path].add(path_cache[import_name])
                        # If importing a module/package directly
                        elif dep_module in module_to_file:
                            dependency_graph[file_path].add(module_to_file[dep_module])

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        dep_module = alias.name
                        if not dep_module:
                            continue

                        if dep_module.startswith(PACKAGE_PREFIX):
                            dep_module = dep_module[len(PACKAGE_PREFIX) :]
                        elif dep_module.startswith('copr.'):
                            dep_module = dep_module[len('copr.') :]

                        if dep_module in module_to_file:
                            dependency_graph[file_path].add(module_to_file[dep_module])
                        else:
                            # Look for modules starting with this prefix
                            for mod_name, mod_path in module_to_file.items():
                                if mod_name == dep_module or mod_name.startswith(
                                    dep_module + '.'
                                ):
                                    dependency_graph[file_path].add(mod_path)
                                    break
        except Exception as e:
            print(f'Error processing {file_path}: {e}')

    # Heuristic: If a package's __init__.py is imported, also add the package file if present
    for file_path in all_files:
        if os.path.basename(file_path) == '__init__.py':
            package_dir = os.path.dirname(file_path)
            package_name = os.path.basename(package_dir)
            candidate = os.path.join(package_dir, f'{package_name}.py')
            if os.path.exists(candidate):
                dependency_graph[file_path].add(candidate)

    # Breadth-first search from initial files
    relevant_files = set(initial_files)
    queue = deque(initial_files)
    while queue:
        current = queue.popleft()
        for dep in dependency_graph.get(current, []):
            if dep in all_files and dep not in relevant_files:
                relevant_files.add(dep)
                queue.append(dep)

    irrelevant_files = set(all_files) - relevant_files
    return {'relevant_files': relevant_files, 'irrelevant_files': irrelevant_files}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Find irrelevant files for a given algorithm in a Python codebase.'
    )
    parser.add_argument(
        'algorithm_name', help="Algorithm name to look for, e.g. 'ppo'."
    )
    parser.add_argument('--base-dir', default='./', help='Base directory to search in.')
    args = parser.parse_args()

    result = find_irrelevant_files(args.algorithm_name, base_dir=args.base_dir)
    print('Relevant files:')
    for r in sorted(result['relevant_files']):
        print('  ', r)

    print('\nIrrelevant files:')
    for i in sorted(result['irrelevant_files']):
        print('  ', i)

    print('\nMake sure you keep the __init__.py files in the relevant directories.')

    # remove all irrelevant files except __init__.py's in the codebase
    for file in result['irrelevant_files']:
        if os.path.basename(file) == '__init__.py':
            continue
        os.remove(file)
