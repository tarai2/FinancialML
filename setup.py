from setuptools import setup

setup(
    name="FinancialML",
    version="0.0.1",
    install_requires = 
    [
        "numpy",
        "pandas",
        "tensorflow",
        "numba"
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