from setuptools import setup
from glob import glob
import os

package_name = 'limo_circle_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'),
            glob('meshes/*')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nipun',
    maintainer_email='nagarwalla@umass.edu',
    description='ROS2 deployment of circle_policy_v1 on LIMO fleet.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'circle_node = limo_circle_sim.circle_node:main',
            'teleop_node = limo_circle_sim.teleop_node:main',
            'markers_node = limo_circle_sim.markers_node:main',
        ],
    },
)
