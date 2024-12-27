#control version

setup(
    name='zedtool',
    version='0.1',
    author='John Markham',
    author_email='john.markham@gmail.com',
    
packages=['zedtool', 'zedtool.detections','zedtool.fiducials','zedtool.plots','zedtool.srxstats','zedtool.image', 'zedtool.cli'],
    url='',
    description='Z Estimate Diagnostics Tool',
    package_dir = {'zedtool':'src/zedtool'},
    package_data = {'zedtool': ['test/*']},
    requires=['NumPy (>=1.6)', 
        "Scipy (>= 0.1)", " Skimage (>= 0.2)"],
)
