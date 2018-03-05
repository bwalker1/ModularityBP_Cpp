from distutils.core import setup, Extension

ext_module=Extension(name="_modbp",
                     sources=[],
                     include_dirs=[]
                     )

options=dict( name='modbp',
    version='0.0',
    packages=['modbp'],
    url='',
    license='GPLv3+',
    author='William Weir',
    provides=['modbp'],
    author_email='wweir@med.unc.edu',
    description='',
    zip_safe=False,
    classifiers=["Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.3",
                 "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                 "Topic :: Scientific/Engineering :: Information Analysis",
                 ],
    install_requires=[]
)
#    install_requires=['pyhull','igraph','louvain','matplotlib','numpy',]

setup(**options)

# setup(name=name, version=version,
#       ext_modules=[Extension(name='_hw',
#       # SWIG requires an underscore as a prefix for the module name
#              sources=["hw.i", "src/hw.c"],
#              include_dirs=['src'])
#     ])