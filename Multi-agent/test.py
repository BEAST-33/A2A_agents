import pkgutil
import google-adk as adk

for loader, name, is_pkg in pkgutil.walk_packages(adk.__path__, adk.__name__ + '.'):
    print(f"{'Package' if is_pkg else 'Module'}: {name}")