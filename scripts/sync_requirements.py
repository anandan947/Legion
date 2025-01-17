#!/usr/bin/env python3
"""Sync dependencies between pyproject.toml, setup.py, and requirements.txt."""
import toml
import re
from pathlib import Path

def read_pyproject_deps():
    """Read dependencies from pyproject.toml."""
    with open('pyproject.toml') as f:
        pyproject = toml.load(f)

    deps = pyproject['tool']['poetry']['dependencies'].items()
    main_deps = {k: v for k, v in deps if k != 'python'}

    dev_deps = pyproject['tool']['poetry']['group']['dev']['dependencies']

    return main_deps, dev_deps

def update_requirements_txt(main_deps, dev_deps):
    """Update requirements.txt with all dependencies."""
    all_deps = []

    for package, version in main_deps.items():
        version = version.replace('^', '')  # Remove poetry's ^ operator
        all_deps.append(f"{package}=={version}")

    for package, version in dev_deps.items():
        version = version.replace('^', '')
        all_deps.append(f"{package}=={version}")

    all_deps.sort()

    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(all_deps) + '\n')

def update_setup_py(main_deps, dev_deps):
    """Update setup.py with dependencies."""
    setup_path = Path('setup.py')
    setup_content = setup_path.read_text()

    main_deps_formatted = [f'        "{package}=={version}",' for package, version in main_deps.items()]
    main_deps_str = '\n'.join(main_deps_formatted)

    dev_deps_formatted = [f'            "{package}={version}",' for package, version in dev_deps.items()]
    dev_deps_str = '\n'.join(dev_deps_formatted)

    setup_content = re.sub(
        r'install_requires=\[(.*?)\]',
        f'install_requires=[\n{main_deps_str}\n    ]',
        setup_content,
        flags=re.DOTALL
    )

    setup_content = re.sub(
        r'"dev":\s*\[(.*?)\]',
        f'"dev": [\n{dev_deps_str}\n        ]',
        setup_content,
        flags=re.DOTALL
    )

    setup_path.write_text(setup_content)

def main():
    """Main function to sync all dependency files."""
    try:
        # Read dependencies from pyproject.toml
        main_deps, dev_deps = read_pyproject_deps()

        # Update requirements.txt
        update_requirements_txt(main_deps, dev_deps)

        # Update setup.py
        update_setup_py(main_deps, dev_deps)

        print("✅ Successfully synced dependencies!")
        return 0
    except Exception as e:
        print(f"❌ Error syncing dependencies: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main())
