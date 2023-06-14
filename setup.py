import setuptools
from os import path
import nmrpred

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='nmrpred',
        version=nmrpred.__version__,
        author='Jie Li',
        author_email='jerry-li1996@berkeley.edu',
        project_urls={
            'Source': 'https://github.com/THGLab/iShiftML',
        },
        description=
        "Highly Accurate Prediction of NMR Chemical Shifts from Low-Level Quantum Mechanics Calculations Using Machine Learning",
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=[
            'Machine Learning', 'NMR chemical shifts', 'active learning',
            'ensemble prediction', "ab initio calculation"
        ],
        license='MIT',
        packages=setuptools.find_packages(),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False,
    )