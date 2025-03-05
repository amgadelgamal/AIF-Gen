import pytest

from aif_gen.util.find_irrelevant_files import (
    find_irrelevant_files,
    get_module_name,
    resolve_relative_import,
)


@pytest.fixture
def test_project(tmpdir):
    # Create a dummy directory structure for testing
    test_dir = tmpdir.mkdir('test_project')
    copr_dir = test_dir.mkdir('copr')
    trlx_dir = copr_dir.mkdir('trlx')
    data_dir = trlx_dir.mkdir('data')
    models_dir = trlx_dir.mkdir('models')
    trainer_dir = trlx_dir.mkdir('trainer')
    utils_dir = trlx_dir.mkdir('utils')

    # Create dummy files with minimal contents.
    # Note: We use imports with "copr." which will be stripped in the code.
    files = {
        copr_dir.join(
            'train_copr.py'
        ): 'import json\nfrom copr.trlx.data.configs import ModelConfig',
        data_dir.join('__init__.py'): '',
        data_dir.join('configs.py'): 'class ModelConfig: pass',
        models_dir.join(
            'modeling_copr.py'
        ): 'from copr.trlx.data.configs import ModelConfig',
        trainer_dir.join(
            'trainer.py'
        ): 'from copr.trlx.models.modeling_copr import ModelConfig',
        utils_dir.join('utils.py'): '',
        utils_dir.join('__init__.py'): '',
        trlx_dir.join('trlx.py'): 'from copr.trlx.utils import utils',
        copr_dir.join('ppo.py'): '',
        copr_dir.join('dpo.py'): '',
    }
    for file_path, content in files.items():
        file_path.write(content)

    return test_dir, files


def test_get_module_name(test_project):
    test_dir, files = test_project
    base_dir = test_dir.join('copr')
    file_path = test_dir.join('copr', 'trlx', 'data', 'configs.py')
    expected_module_name = 'trlx.data.configs'
    assert get_module_name(str(file_path), str(base_dir)) == expected_module_name

    file_path = test_dir.join('copr', 'trlx', 'data', '__init__.py')
    expected_module_name = 'trlx.data'
    assert get_module_name(str(file_path), str(base_dir)) == expected_module_name


def test_resolve_relative_import():
    this_module = 'copr.trlx.trainer.trainer'
    level = 2
    imported_module = 'data.configs'
    # Removing 2 parts from 'copr.trlx.trainer.trainer' yields 'copr.trlx',
    # then appending "data.configs" gives 'copr.trlx.data.configs'
    expected_module = 'copr.trlx.data.configs'
    assert (
        resolve_relative_import(this_module, level, imported_module) == expected_module
    )

    this_module = 'copr.trlx.trainer'
    level = 1
    imported_module = 'utils'
    expected_module = 'copr.trlx.utils'
    assert (
        resolve_relative_import(this_module, level, imported_module) == expected_module
    )


def test_find_irrelevant_files_copr(test_project):
    test_dir, files = test_project
    # For algorithm "copr", initial files are those with "copr" in the basename.
    # Here that is "train_copr.py" and "modeling_copr.py".
    # Each of these imports "from copr.trlx.data.configs", which, after stripping, resolves to "trlx.data.configs"
    # so that file gets added.
    relevant = {
        str(test_dir.join('copr', 'train_copr.py')),
        str(test_dir.join('copr', 'trlx', 'models', 'modeling_copr.py')),
        str(test_dir.join('copr', 'trlx', 'data', 'configs.py')),
    }
    # All other Python files become irrelevant.
    irrelevant = {
        str(test_dir.join('copr', 'trlx', 'data', '__init__.py')),
        str(test_dir.join('copr', 'trlx', 'trainer', 'trainer.py')),
        str(test_dir.join('copr', 'trlx', 'utils', 'utils.py')),
        str(test_dir.join('copr', 'trlx', 'utils', '__init__.py')),
        str(test_dir.join('copr', 'trlx', 'trlx.py')),
        str(test_dir.join('copr', 'ppo.py')),
        str(test_dir.join('copr', 'dpo.py')),
    }
    result = find_irrelevant_files('copr', base_dir=str(test_dir.join('copr')))
    assert set(result['relevant_files']) == relevant
    assert set(result['irrelevant_files']) == irrelevant


def test_find_irrelevant_files_ppo(test_project):
    test_dir, files = test_project
    # For algorithm "ppo", only "ppo.py" is initial
    relevant = {
        str(test_dir.join('copr', 'ppo.py')),
    }
    irrelevant = {
        str(test_dir.join('copr', 'train_copr.py')),
        str(test_dir.join('copr', 'trlx', 'data', 'configs.py')),
        str(test_dir.join('copr', 'trlx', 'models', 'modeling_copr.py')),
        str(test_dir.join('copr', 'trlx', 'trainer', 'trainer.py')),
        str(test_dir.join('copr', 'trlx', 'utils', 'utils.py')),
        str(test_dir.join('copr', 'trlx', 'utils', '__init__.py')),
        str(test_dir.join('copr', 'trlx', 'data', '__init__.py')),
        str(test_dir.join('copr', 'trlx', 'trlx.py')),
        str(test_dir.join('copr', 'dpo.py')),
    }
    result = find_irrelevant_files('ppo', base_dir=str(test_dir.join('copr')))
    assert set(result['relevant_files']) == relevant
    assert set(result['irrelevant_files']) == irrelevant


def test_find_irrelevant_files_no_algorithm(test_project):
    test_dir, files = test_project
    # When no file contains the algorithm name, the relevant set should be empty.
    result = find_irrelevant_files('nonexistent', base_dir=str(test_dir.join('copr')))
    assert set(result['relevant_files']) == set()
    assert set(result['irrelevant_files']) == {str(f) for f in files.keys()}


def test_package_heuristic(test_project):
    test_dir, files = test_project
    # Create a dummy package structure in copr/test_package.
    test_package_dir = test_dir.join('copr').mkdir('test_package')
    test_package_dir.join('__init__.py').write('')
    test_package_file = test_package_dir.join('test_package.py')
    test_package_file.write('')
    main_file = test_dir.join('copr').join('main.py')
    main_file.write('import copr.test_package')

    # Update the files dict.
    files[test_package_dir.join('__init__.py')] = ''
    files[test_package_file] = ''
    files[main_file] = 'import copr.test_package'

    # For "main", initial file is "main.py". Its import "copr.test_package"
    # is stripped to "test_package" and then resolved to the package's __init__.py.
    # The heuristic then adds the package's main file, i.e. test_package.py.
    relevant = {
        str(main_file),
        # Explicit import of a package should include its __init__.py
        str(test_dir.join('copr', 'test_package', '__init__.py')),
        str(test_dir.join('copr', 'test_package', 'test_package.py')),
    }
    irrelevant = set(str(f) for f in files.keys()) - relevant

    result = find_irrelevant_files('main', base_dir=str(test_dir.join('copr')))
    assert set(result['relevant_files']) == relevant
    assert set(result['irrelevant_files']) == irrelevant


def test_nested_relative_imports(test_project):
    """Test handling of nested relative imports."""
    test_dir, files = test_project

    # Create a more complex structure with nested modules
    nested_dir = test_dir.join('copr').mkdir('nested')
    sub1_dir = nested_dir.mkdir('sub1')
    sub2_dir = sub1_dir.mkdir('sub2')

    # Create files with relative imports
    nested_dir.join('__init__.py').write('')
    sub1_dir.join('__init__.py').write('')
    sub2_dir.join('__init__.py').write('')

    # Main implementation file
    main_file = nested_dir.join('nested_impl.py')
    main_file.write('from .sub1.sub2 import helper')

    # A helper module in the nested structure
    helper_file = sub2_dir.join('helper.py')
    helper_file.write('# Helper functions\ndef some_helper(): pass')

    # Another file with deeper relative imports
    user_file = sub1_dir.join('user.py')
    user_file.write('from .. import nested_impl\nfrom .sub2.helper import some_helper')

    # Update files dictionary
    files[nested_dir.join('__init__.py')] = ''
    files[sub1_dir.join('__init__.py')] = ''
    files[sub2_dir.join('__init__.py')] = ''
    files[main_file] = 'from .sub1.sub2 import helper'
    files[helper_file] = '# Helper functions\ndef some_helper(): pass'
    files[user_file] = (
        'from .. import nested_impl\nfrom .sub2.helper import some_helper'
    )

    # Run the test with algo "nested"
    result = find_irrelevant_files('nested', base_dir=str(test_dir.join('copr')))

    # Verify the dependencies are correctly resolved
    relevant = {
        str(main_file),  # The initial file with "nested" in name
        str(helper_file),  # Imported by main_file
    }

    assert set(result['relevant_files']) == relevant
    assert (
        str(user_file) in result['irrelevant_files']
    )  # user.py isn't reached by traversal


def test_recursive_dependencies(test_project):
    """Test recursive chain of dependencies A -> B -> C -> D."""
    test_dir, files = test_project

    # Create a chain of files with dependencies
    rec_dir = test_dir.join('copr').mkdir('recursive')
    rec_dir.join('__init__.py').write('')

    # Create files with a chain of dependencies
    file_a = rec_dir.join('rec_a.py')
    file_a.write('import copr.recursive.b')

    file_b = rec_dir.join('b.py')
    file_b.write('from copr.recursive.c import helper')

    file_c = rec_dir.join('c.py')
    file_c.write('from copr.recursive.d import final_function')

    file_d = rec_dir.join('d.py')
    file_d.write('def final_function(): return "result"')

    # Update files dictionary
    files[rec_dir.join('__init__.py')] = ''
    files[file_a] = 'import copr.recursive.b'
    files[file_b] = 'from copr.recursive.c import helper'
    files[file_c] = 'from copr.recursive.d import final_function'
    files[file_d] = 'def final_function(): return "result"'

    # Run the test with algo "rec"
    result = find_irrelevant_files('rec', base_dir=str(test_dir.join('copr')))

    # Verify all dependencies are included
    relevant = {
        str(file_a),  # The initial file with "rec" in name
        str(file_b),  # Imported by file_a
        str(file_c),  # Imported by file_b
        str(file_d),  # Imported by file_c
        # NOTE: No longer including __init__.py in relevant files for consistency
    }

    assert set(result['relevant_files']) == relevant


def test_circular_dependencies(test_project):
    """Test handling of circular dependencies."""
    test_dir, files = test_project

    # Create files with circular dependencies
    circ_dir = test_dir.join('copr').mkdir('circular')
    circ_dir.join('__init__.py').write('')

    # Create files that import each other
    file_a = circ_dir.join('circ_a.py')
    file_a.write('import copr.circular.b\ndef func_a(): pass')

    file_b = circ_dir.join('b.py')
    file_b.write('import copr.circular.circ_a\ndef func_b(): pass')

    # Update files dictionary
    files[circ_dir.join('__init__.py')] = ''
    files[file_a] = 'import copr.circular.b\ndef func_a(): pass'
    files[file_b] = 'import copr.circular.circ_a\ndef func_b(): pass'

    # Run the test with algo "circ"
    result = find_irrelevant_files('circ', base_dir=str(test_dir.join('copr')))

    # Verify both sides of the circular dependency are included
    relevant = {
        str(file_a),  # The initial file with "circ" in name
        str(file_b),  # Imported by file_a and imports file_a
        # NOTE: No longer including __init__.py in relevant files for consistency
    }

    assert set(result['relevant_files']) == relevant


def test_complex_package_structure(test_project):
    """Test handling of complex package structures with many subpackages."""
    test_dir, files = test_project

    # Create a complex package structure
    complex_dir = test_dir.join('copr').mkdir('complex')
    complex_dir.join('__init__.py').write('')

    # Subpackages
    api_dir = complex_dir.mkdir('api')
    api_dir.join('__init__.py').write('')

    models_dir = complex_dir.mkdir('models')
    models_dir.join('__init__.py').write('')

    utils_dir = complex_dir.mkdir('utils')
    utils_dir.join('__init__.py').write('')

    # Config is imported by multiple other modules
    config_file = complex_dir.join('config.py')
    config_file.write('CONFIG = {"setting": "value"}')

    # API module imports from models and utils
    api_impl = api_dir.join('impl.py')
    api_impl.write(
        'from copr.complex.models import user\nfrom copr.complex.utils.helpers import format_response'
    )

    # Models module
    user_model = models_dir.join('user.py')
    user_model.write('from copr.complex.config import CONFIG')

    # Utils module also uses config
    helpers = utils_dir.join('helpers.py')
    helpers.write(
        'from copr.complex.config import CONFIG\n\ndef format_response(data): return str(data)'
    )

    # Main module that ties it all together
    main = complex_dir.join('complex_app.py')
    main.write('from copr.complex.api.impl import process_request')

    # Update files dictionary
    files[complex_dir.join('__init__.py')] = ''
    files[api_dir.join('__init__.py')] = ''
    files[models_dir.join('__init__.py')] = ''
    files[utils_dir.join('__init__.py')] = ''
    files[config_file] = 'CONFIG = {"setting": "value"}'
    files[api_impl] = (
        'from copr.complex.models import user\nfrom copr.complex.utils.helpers import format_response'
    )
    files[user_model] = 'from copr.complex.config import CONFIG'
    files[helpers] = (
        'from copr.complex.config import CONFIG\n\ndef format_response(data): return str(data)'
    )
    files[main] = 'from copr.complex.api.impl import process_request'

    # Run the test with algo "complex"
    result = find_irrelevant_files('complex', base_dir=str(test_dir.join('copr')))

    # All these files should be relevant due to dependencies
    relevant = {
        str(main),  # The initial file with "complex" in name
        str(api_impl),
        str(user_model),
        str(helpers),
        str(config_file),
        # NOTE: No longer including __init__.py files in relevant files for consistency
    }

    assert set(result['relevant_files']) == relevant
