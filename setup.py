from setuptools import setup

setup(
    name="financialml",
    version="0.0.1",
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "numba",
        "pytest",
        "pytest-mock"
    ]
    # extras_require =
    # {
    #     "develop": ["dev-packageA", "dev-packageB"]
    # },

    # entry_points =
    # {
    #     "console_scripts" :
    #     [
    #         "foo = package_name.module_name:func_name",
    #         "foo_dev = package_name.module_name:func_name [develop]"
    #     ]
    # }
)
