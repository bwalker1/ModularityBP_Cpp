#from distutils.core import setup, Extension

from setuptools import setup, Extension
import os,re
ext_module=Extension(name="_bp",
                    sources=["modbp/src_cpp/bp.cpp","modbp/bp.i"],
                    include_dirs=["modbp/src_cpp"],swig_opts=["-c++"],extra_compile_args=['-std=c++11'],
                    )

#setup version information
#read inversion info from single file
PKG = "modbp"
VERSIONFILE = os.path.join(PKG, "_version.py")
verstr = "unknown"
try:
    verstrline = open(VERSIONFILE, "rt").read()
    print(verstrline)
except EnvironmentError:
    pass # Okay, there is no version file.
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        print ("unable to find version in %s" % (VERSIONFILE,))
        raise RuntimeError("if %s.py exists, it is required to be well-formed" % (VERSIONFILE,))


options=dict( name=PKG,
    version=verstr,
    packages=[PKG],
    url='',
    license='GPLv3+',
    author='William Weir',
    provides=['modbp'],
    author_email='wweir@med.unc.edu',
    description='This is a community detection package that used a belief propagation approach to optimize modularity on multilayer networks.  Algorithm is implemented in c++ with python interface for convenience. ',
    ext_modules=[ext_module],
    zip_safe=False,
    classifiers=["Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.6",
                 "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                 "Topic :: Scientific/Engineering :: Information Analysis",
                 ],
    install_requires=['python-igraph','numpy','matplotlib','scipy','scikit-learn','seaborn','pandas']
)
#    install_requires=['pyhull','igraph','louvain','matplotlib','numpy',]

setup(**options)

# setup(name=name, version=version,
#       ext_modules=[Extension(name='_hw',
#       # SWIG requires an underscore as a prefix for the module name
#              sources=["hw.i", "src/hw.c"],
#              include_dirs=['src'])
#     ])
