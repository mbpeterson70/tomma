from setuptools import setup

setup(
    name='casadi_trajectory_optimization',
    version='0.1.0',    
    description='Trajectory optimization using Casadi',
    url='https://github.com/mbpeterson70/casadi_trajectory_optimization',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    packages=['casadi_trajectory_optimization'],
    install_requires=['numpy',
                        'matplotlib',
                        'ipython',
                        'sympy',
                        'scipy',
                        'casadi',
                        'ipywidgets',
                      ],

    # classifiers=[
    #     'Development Status :: 1 - Planning',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',  
    #     'Operating System :: POSIX :: Linux',        
    #     'Programming Language :: Python :: 2',
    #     'Programming Language :: Python :: 2.7',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.4',
    #     'Programming Language :: Python :: 3.5',
    # ],
)