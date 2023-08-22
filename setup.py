from setuptools import setup

setup(
    name='tomma',
    version='0.1.0',    
    description='Trajectory Optimization for Multiple Model-based Agents',
    url='https://github.com/mbpeterson70/casadi_trajectory_optimization',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    packages=['tomma'],
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