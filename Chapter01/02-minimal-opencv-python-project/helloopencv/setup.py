from setuptools import setup

setup(
    name='helloopencv',
    version='0.1',
    py_modules=["helloopencv"],
    license='MIT',
    description='An example python opencv project',
    long_description=open('README.rst').read(),
    install_requires=['numpy','opencv-contrib-python'],
    url='https://github.com/albertofernandez',
    author='Alberto Fernandez',
    author_email='fernandezvillan.alberto@gmail.com'
)
